# read_lines.py: for every photo in data/, splits it into lines (segment.py),
# reads each line with TrOCR twice: normally (then cleanup.py) and restricted to math characters,
# records TrOCR's confidence,
# checks whether each step follows from the previous one (step_check.py),
# and saves everything to results.csv
# usage: put all photos in data/ (e.g. 1.jpg, 2.jpg, ... 20.jpg), then run: python read_lines.py

import glob  # python library for finding files that match a pattern
import re  # for sorting line_2 before line_10
import csv  # for saving results.csv
import os
import subprocess, sys  # for running segment.py from here
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from cleanup import clean  # our function to fix common mistakes and remove junk
from step_check import check_solution  # math consistency check between steps

# load pretrained TrOCR model and processor (only once, for all photos)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model.config.vocab_size = model.config.decoder.vocab_size  # fix: needed for compute_transition_scores on older transformers

# NEW: list of tokens TrOCR is allowed to output in "math only" mode
MATH_CHARS = set("0123456789xX+-=/()^ ")
tok = processor.tokenizer
allowed_ids = [i for i in range(len(tok))
               if (t := tok.decode([i])).strip() and set(t) <= MATH_CHARS]
allowed_ids.append(tok.eos_token_id)  # so it can stop


def only_math(batch_id, input_ids):
    return allowed_ids  # at every step, only math tokens can be picked


def read_line(pixel_values, math_only=False):
    """read one line image with TrOCR; returns text, mean confidence, lowest confidence"""
    out = model.generate(pixel_values, max_new_tokens=50, output_scores=True, return_dict_in_generate=True,
                         prefix_allowed_tokens_fn=only_math if math_only else None)
    text = processor.batch_decode(out.sequences, skip_special_tokens=True)[0]  # token ids into text
    scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
    probs = scores[0].exp()  # how confident it was about each token (0 to 1)
    return text, probs.mean().item(), probs.min().item()


def number_in(path):
    """pull the number out of a name, so sol2 sorts before sol10"""
    nums = re.findall(r"\d+", os.path.basename(path))
    return int(nums[-1]) if nums else 0


# every photo in data/ (jpg, jpeg or png)
photos = sorted(
    [p for p in glob.glob("data/*") if p.lower().endswith((".jpg", ".jpeg", ".png"))],
    key=number_in,
)

with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["photo", "line", "raw", "cleaned", "math_only", "conf", "lowest", "step_check"])

    for photo in photos:
        name = os.path.basename(photo)

        # split this photo into output/line_0.jpg, line_1.jpg, ...
        subprocess.run([sys.executable, "segment.py", photo], check=True)

        line_paths = sorted(glob.glob("output/line_*.jpg"), key=number_in)
        raws, cleaned_lines, math_lines, confs, lowest_confs = [], [], [], [], []

        for path in line_paths:
            img = Image.open(path).convert("RGB")
            pixel_values = processor(images=img, return_tensors="pt").pixel_values  # turn image into numbers

            # 1) normal TrOCR + cleanup.py (baseline, kept for comparison)
            text, _, _ = read_line(pixel_values)
            raws.append(text)
            cleaned_lines.append(clean(text))

            # 2) NEW: TrOCR only allowed to output math characters
            math_text, conf, lowest = read_line(pixel_values, math_only=True)
            math_lines.append(math_text.replace(" ", "").replace("X", "x"))
            confs.append(conf)  # mean confidence for the whole line
            lowest_confs.append(lowest)  # confidence of its least confident token

        # check each step against the previous one (using the math-only reading)
        labels = check_solution(math_lines, lowest_confs)

        print(f"\n{name}:")
        for i in range(len(line_paths)):
            print(f"  raw: {raws[i]} | cleaned: {cleaned_lines[i]} | math only: {math_lines[i]} "
                  f"| conf {confs[i]:.2f} | lowest {lowest_confs[i]:.2f} | {labels[i]}")
            writer.writerow([name, i, raws[i], cleaned_lines[i], math_lines[i],
                             f"{confs[i]:.2f}", f"{lowest_confs[i]:.2f}", labels[i]])

print("\nsaved results.csv")