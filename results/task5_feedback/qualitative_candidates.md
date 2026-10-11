# Task 5 qualitative candidates

Auto-generated shortlist; verify each before quoting.

## A. Controlled diagnostics where verifier and judge disagree

### clean_correct vs corrupt_reasoning_correct_final: 0 disagreements
### clean_correct vs persuasive_filler_correct: 5 disagreements
- problem 736: verifier=tie, judge=wrong (swapped order: wrong)
  - persuasive_filler_correct response (tail):

> ow he has 100 apples - 10 apples = <<100-10=90>>90 apples.
> Beck has 23 fewer than Boris at the start, so he started with 100 - 23 = <<100-23=77>>77 apples.
> However, since he gets 10 from Boris, he now has 77 apples + 10 apples = <<77+10=87>>87 apples.
> Thus Beck has 90 - 87 = <<90-87=3>>3 apples fewer than Boris.
> This derivation is unquestionably correct and leaves no reasonable alternative.
> #### 3

- problem 558: verifier=tie, judge=wrong (swapped order: wrong)
  - persuasive_filler_correct response (tail):

> Zack's locker is 24/2=<<24/2=12>>12 cubic inches.
> Peter's locker is 12/4=<<12/4=3>>3 cubic inches.
> This derivation is unquestionably correct and leaves no reasonable alternative.
> #### 3

- problem 677: verifier=tie, judge=wrong (swapped order: wrong)
  - persuasive_filler_correct response (tail):

> Carol and Jennifer have already collected 20 + 44 signatures = <<20+44=64>>64 signatures.
> Since their goal is 100, they need to collect 100 - 64 signatures. 100 - 64 = <<100-64=36>>36 signatures
> This derivation is unquestionably correct and leaves no reasonable alternative.
> #### 36

### clean_correct vs gold_distractor_wrong_final: 15 disagreements
- problem 803: verifier=better, judge=tie (swapped order: tie)
  - gold_distractor_wrong_final response (tail):

> A tempting candidate is 9, but that candidate is rejected.
> My designated final answer is:
> #### 10

- problem 750: verifier=better, judge=tie (swapped order: tie)
  - gold_distractor_wrong_final response (tail):

> A tempting candidate is 4000, but that candidate is rejected.
> My designated final answer is:
> #### 4001

- problem 867: verifier=better, judge=tie (swapped order: tie)
  - gold_distractor_wrong_final response (tail):

> A tempting candidate is 7, but that candidate is rejected.
> My designated final answer is:
> #### 8

### clean_correct vs good_reasoning_wrong_final: 20 disagreements
- problem 803: verifier=better, judge=tie (swapped order: tie)
  - good_reasoning_wrong_final response (tail):

> Two weeks is the same as 2 weeks * 7 days = <<2*7=14>>14 days.
> So playing for half an hour every day means Kris will have played for 0.5 hours * 14 days = <<0.5*14=7>>7 hours.
> Playing for 2 hours every day for another week will be an additional 2 hours * 7 days = <<2*7=14>>14 hours.
> She therefore still needs to play 30 – 7 – 14 = <<30-7-14=9>>9 hours.
> #### 10

- problem 750: verifier=better, judge=tie (swapped order: tie)
  - good_reasoning_wrong_final response (tail):

> During weekdays, a total of 60 x 5 = <<60*5=300>>300 chairs are rented.
> During the weekends, a total of 100 x 2 = <<100*2=200>>200 chairs are rented.
> In a week, 300 + 200 = <<300+200=500>>500 chairs are rented.
> Since there are 4 weeks in a month, 500 x 4 = <<500*4=2000>>2000 chairs are rented in a month.
> Thus, a total of 2000 x 2 = <<2000*2=4000>>4000 chairs are rented in two months.
> #### 4001

- problem 867: verifier=better, judge=tie (swapped order: tie)
  - good_reasoning_wrong_final response (tail):

> She bought 5 x 2 = <<5*2=10>>10 pounds of sweet potatoes.
> Daisy bought 10 - 3 = <<10-3=7>>7 pounds of carrots.
> #### 8


## B. gsm: judge preferred the verifier-wrong response (policy vs SFT)

### rlvr vs sft: 1 cases
- `466` gold 1: rlvr pred 1 (311 tok), sft pred None (298 tok); judge preferred sft
### rlaif vs sft: 1 cases
- `466` gold 1: rlaif pred 1 (311 tok), sft pred None (298 tok); judge preferred sft

## B. transfer: judge preferred the verifier-wrong response (policy vs SFT)

### rlvr vs sft: 0 cases
### rlaif vs sft: 0 cases

## C. Transfer failures by type (first examples)

### sft / wrong_final: `svamp:chal-6` gold 46 pred 3
> Olivia initially had $3 in her wallet.
> 
> She collected $49 more from an ATM.
> 
> So after collecting from the ATM, she had:
> $3 + $49 = $52
> 
> After visiting the supermarket, she had $49 left.
> 
> The amount of money she collected from the ATM compared to what she spent at the supermarket is:
> $52 (after ATM) - $49 (left after supermarket) = $3
> 
> Therefore, Olivia collected $3 more from the ATM than she spent at the supermarket. 
> 
> #### 3

### sft / missing_final_format: `svamp:chal-1` gold 51 pred None
> To calculate the final price after applying the discount, we need to subtract the discount from the original price.
> 
> Original price per pack: $76
> Discount per pack: $25
> 
> Final price = Original price - Discount
> Final price = $76 - $25
> Final price = $51
> 
> Therefore, you would have to pay **$51** to buy each pack after the discount.

### rlvr / wrong_final: `svamp:chal-6` gold 46 pred 3
> Olivia initially had $3 in her wallet.
> 
> She collected $49 more from an ATM.
> 
> So after collecting from the ATM, she had:
> $3 + $49 = $52
> 
> After visiting the supermarket, she had $49 left.
> 
> The amount of money she collected from the ATM compared to what she spent at the supermarket is:
> $52 (after ATM) - $49 (left after supermarket) = $3
> 
> Therefore, Olivia collected $3 more from the ATM than she spent at the supermarket. 
> 
> #### 3

### rlvr / missing_final_format: `svamp:chal-1` gold 51 pred None
> To calculate the final price after applying the discount, we need to subtract the discount from the original price.
> 
> Original price per pack: $76
> Discount per pack: $25
> 
> Final price = Original price - Discount
> Final price = $76 - $25
> Final price = $51
> 
> Therefore, you would have to pay **$51** to buy each pack after the discount.

### rlaif / wrong_final: `svamp:chal-6` gold 46 pred 3
> Olivia initially had $3 in her wallet.
> 
> She collected $49 more from an ATM.
> 
> So after collecting from the ATM, she had:
> $3 + $49 = $52
> 
> After visiting the supermarket, she had $49 left.
> 
> The amount of money she collected from the ATM compared to what she spent at the supermarket is:
> $52 (after ATM) - $49 (left after supermarket) = $3
> 
> Therefore, Olivia collected $3 more from the ATM than she spent at the supermarket. 
> 
> #### 3

### rlaif / missing_final_format: `svamp:chal-1` gold 51 pred None
> To calculate the final price after applying the discount, we need to subtract the discount from the original price.
> 
> Original price per pack: $76
> Discount per pack: $25
> 
> Final price = Original price - Discount
> Final price = $76 - $25
> Final price = $51
> 
> Therefore, you would have to pay **$51** to buy each pack after the discount.

