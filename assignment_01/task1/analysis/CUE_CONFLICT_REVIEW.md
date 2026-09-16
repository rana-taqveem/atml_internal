# Cue-conflict generation and visual-review protocol

Define this protocol before production review and before evaluating classifiers. Pilot images helped choose settings; they are not automatically accepted production examples. This protocol is a design choice, not additional assignment wording.

## Generation plan

- Use the five configured unordered class pairs, in both directions: ten groups.
- Initial pool: 40 candidates per direction, 400 total. Target: 20 visually valid selected images per direction, 200 total.
- Use seed 6304, fixed alpha=1.0 for this proposed production run, and the saved 500-image test subset. Record the actual setting in every record.
- Randomly order distinct content/style combinations without replacement within each direction. Save that order; never resample in response to predictions.
- Generate candidates once; save individual PNGs, content/style/output previews and metadata. All evaluated methods load the same selected PNGs.
- Forty candidates do not guarantee twenty valid outputs. If a group is short, continue from its saved unused combination order, review new candidates and record them. Do not relax rules only for difficult pairs. Document any unavoidable imbalance.

## Visual acceptance rule v1

Review content, style and output together without model predictions. Accept only if all conditions hold:

1. The source content and style objects are visually interpretable; the style example contains discernible surface structure.
2. The output preserves a recognizable content object and its overall geometry.
3. The output visibly incorporates surface pattern/texture from the style example on the content object; a color shift alone is insufficient. Literal identifiable feathers or species recognition are not required.
4. The image is not dominated by severe blur, saturation, fragmentation or background-only transfer that makes the intended cue conflict ambiguous.

Reject if any condition fails or remains uncertain. Suggested reason codes: source_ambiguous, shape_lost, weak_texture, background_dominates, severe_artifacts, ambiguous. Record a short explanation; multiple reasons may apply. Do not choose candidates based on predictions or subsequent shape-bias scores.

## Review records and selection

Each record contains conflict_id, pair, direction, candidate order, original content/style IDs and labels, alpha, seed, image/preview paths, accepted (null until reviewed), rejection_reason and selected_for_evaluation (initially false).

Review every generated candidate to distinguish accepted, rejected and unreviewed counts. Select the first twenty accepted examples per direction in the saved random order. Extra accepted candidates remain accepted but unselected: they are not rejected. Report generated, reviewed, accepted, rejected, unreviewed and selected counts, overall and by pair/direction. With all candidates reviewed, generated = accepted + rejected; selected <= accepted.

## Scope and limitations

Automate sampling, generation, saving and count summaries. Human review establishes visual validity; a large pixel/feature change alone does not establish transferred object texture. Selection constrains conclusions to visually valid stylizations; report rejection patterns rather than hiding difficult groups.

Status: protocol documented; production generation/review not yet executed or verified. Python implementation is supplied in chat for manual typing.
