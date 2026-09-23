"""PROSER: classifier placeholders and data placeholders.

Zhou et al. (2021), "Learning Placeholders for Open-Set Recognition".

The problem PROSER attacks: a closed-set classifier has no notion of "none of
these", and no unknown examples are available to teach it one. PROSER
manufactures both a place to put unknowns and examples that belong there,
using only CIFAR-10 data.

Two placeholders, trained on two halves of every mini-batch.

1. Classifier placeholders. Append C_dummy extra output units. For a labelled
   example the correct known class must still win, but with that class removed
   from consideration a dummy must be the strongest remaining response. So the
   dummies learn to sit just outside each known class rather than anywhere:

       L_classifier = CE(z_all, y) + beta * CE(z_with_y_masked, dummy_target)

   The first term preserves closed-set accuracy; the second reserves the
   runner-up position for the dummies.

2. Data placeholders. There are no real unknowns, so PROSER synthesizes proxy
   ones with manifold mixup: interpolate the layer2 activations of two
   examples from *different* classes, push the blend through the rest of the
   network, and train it to be classified as a dummy.

       h~ = lam * h_i + (1 - lam) * h_j,   lam ~ Beta(2, 2),  y_i != y_j
       L_data = gamma * CE(classifier(post(h~)), dummy_target)

   A point between two known classes is a place a confident known prediction
   should not be made, so these act as stand-in unknowns and tighten the
   boundaries between known regions.

Neither objective uses a CIFAR-100 image.

Scoring. The known-class logits keep their original meaning, so MLS over the
ten known logits is directly comparable with Vanilla and GCSC. The
placeholder-based score instead compares the strongest dummy against the
strongest known response, which is the reference implementation's rule:

    u_placeholder(x) = max_d z_dummy_d - max_k z_known_k

Both are reported. Closed-set accuracy always uses the ten known logits only,
so classification and rejection stay separable.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from assignment_01.task4.config import task_config


class ProserNet(nn.Module):
    """Wraps a trained CifarResNet18 and appends dummy classifier units.

    The known classifier is reused unchanged, so the model starts as an exact
    copy of the Vanilla checkpoint and the dummy units begin random. Logits are
    returned as one [N, 10 + C_dummy] tensor whose first ten columns keep their
    original meaning.
    """

    def __init__(self, backbone, num_dummy=None):
        super().__init__()
        self.backbone = backbone
        self.num_dummy = num_dummy or task_config.PROSER_NUM_DUMMY
        self.num_known = backbone.num_classes
        self.dummy_classifier = nn.Linear(backbone.feature_dim, self.num_dummy)

    def logits_from_features(self, features):
        """features [N, 512] -> [N, 10 + C_dummy]."""
        known = self.backbone.classifier(features)      # [N, 10]
        dummy = self.dummy_classifier(features)         # [N, C_dummy]
        return torch.cat([known, dummy], dim=1)

    def forward(self, x, return_features=False):
        features = self.backbone.forward_features(x)    # [N, 512]
        logits = self.logits_from_features(features)    # [N, 10 + C_dummy]
        if return_features:
            return logits, features
        return logits

    def forward_mixed(self, x, permutation, lam):
        """Manifold mixup after layer2, before layer3.

        x            [N, 3, 32, 32]
        permutation  [N]     pairing index, giving example j for each i
        lam          scalar in [0, 1]
        returns      [N, 10 + C_dummy] for the interpolated activations
        """
        h = self.backbone.forward_pre(x)                # [N, 128, 16, 16]
        mixed = lam * h + (1.0 - lam) * h[permutation]  # [N, 128, 16, 16]
        features = self.backbone.forward_post(mixed)    # [N, 512]
        return self.logits_from_features(features)


def classifier_placeholder_loss(logits, labels, num_known, beta=None):
    """Known class stays largest; with it masked out, a dummy must win.

    logits   [N, 10 + C_dummy]
    labels   [N] in 0..9
    returns  scalar

    The second term sets the true class logit to -inf and then asks for the
    best dummy. Because every dummy column remains available, cross-entropy
    against "the dummy block" is computed by treating the masked problem as a
    (10 + C_dummy)-way task whose target is the strongest dummy - which is
    what the reference implementation does.
    """
    beta = task_config.PROSER_BETA if beta is None else beta

    closed_set = F.cross_entropy(logits, labels)

    masked = logits.clone()
    masked.scatter_(1, labels.view(-1, 1), float("-inf"))   # remove the true class
    # Target: whichever dummy currently responds most strongly.
    best_dummy = masked[:, num_known:].argmax(dim=1) + num_known   # [N]
    placeholder = F.cross_entropy(masked, best_dummy)

    return closed_set + beta * placeholder


def data_placeholder_loss(mixed_logits, num_known, gamma=None):
    """Interpolated between-class points should be classified as dummies.

    mixed_logits  [N, 10 + C_dummy]
    returns       scalar
    """
    gamma = task_config.PROSER_GAMMA if gamma is None else gamma
    best_dummy = mixed_logits[:, num_known:].argmax(dim=1) + num_known
    return gamma * F.cross_entropy(mixed_logits, best_dummy)


def different_class_permutation(labels, generator=None):
    """A pairing in which no example is matched with its own class.

    labels   [N]
    returns  [N] index tensor, and a [N] bool mask of usable pairs

    Manifold mixup for data placeholders is only meaningful between different
    classes: blending two dogs produces a dog, not a proxy unknown. A single
    random roll of the batch pairs most examples with a different class; the
    few same-class collisions are dropped rather than resampled, which keeps
    the step deterministic under a fixed seed.
    """
    batch_size = labels.size(0)
    if generator is None:
        permutation = torch.randperm(batch_size, device=labels.device)
    else:
        permutation = torch.randperm(batch_size, generator=generator).to(labels.device)
    usable = labels != labels[permutation]
    return permutation, usable


def placeholder_detection_score(logits, num_known):
    """u = max dummy logit - max known logit.

    logits   [N, 10 + C_dummy] as numpy
    returns  [N]

    Positive when the dummies respond more strongly than any known class, so
    larger still means more novel and the shared thresholding rule applies
    unchanged.
    """
    known = logits[:, :num_known].max(axis=1)
    dummy = logits[:, num_known:].max(axis=1)
    return dummy - known


def known_logits(logits, num_known):
    """The ten known columns, for CSA and for MLS comparable with Vanilla."""
    return logits[:, :num_known]


def train_one_epoch(model, loader, optimizer, device=None, beta=None, gamma=None):
    """One PROSER epoch: each mini-batch split in half between the objectives.

    The assignment requires the split, so the first half trains classifier
    placeholders on real examples and the second half trains data placeholders
    on manifold-mixup blends.
    """
    device = device or task_config.DEVICE
    model.train()

    totals = {"loss": 0.0, "classifier": 0.0, "data": 0.0}
    correct = 0
    seen = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        half = images.size(0) // 2
        if half < 2:
            continue

        first_images, first_labels = images[:half], labels[:half]
        second_images, second_labels = images[half:], labels[half:]

        optimizer.zero_grad(set_to_none=True)

        # Half 1: classifier placeholders on real examples.
        logits = model(first_images)                        # [half, 10 + C_dummy]
        classifier_loss = classifier_placeholder_loss(
            logits, first_labels, model.num_known, beta=beta)

        # Half 2: data placeholders on between-class manifold mixup.
        permutation, usable = different_class_permutation(second_labels)
        if usable.any():
            lam = float(np.random.beta(task_config.PROSER_MIXUP_ALPHA,
                                       task_config.PROSER_MIXUP_ALPHA))
            mixed_logits = model.forward_mixed(second_images, permutation, lam)
            data_loss = data_placeholder_loss(
                mixed_logits[usable], model.num_known, gamma=gamma)
        else:
            data_loss = logits.new_zeros(())

        loss = classifier_loss + data_loss
        loss.backward()
        optimizer.step()

        batch = first_images.size(0)
        totals["loss"] += loss.item() * batch
        totals["classifier"] += classifier_loss.item() * batch
        totals["data"] += float(data_loss) * batch
        correct += (logits[:, :model.num_known].argmax(1) == first_labels).sum().item()
        seen += batch

    result = {key: value / max(seen, 1) for key, value in totals.items()}
    result["accuracy"] = 100.0 * correct / max(seen, 1)
    return result
