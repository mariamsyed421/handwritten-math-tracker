# read_lines.py: Reads each line image (created by segment.py) with TrOCR
# cleans the text (cleanup.py) and prints how confident TrOCR was about each line

import glob #python library for finding files that match a pattern 
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from cleanup import clean  # our function to fix common mistakes and remove junk

# load pretrained TrOCR model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# go through line_0.jpg, line_1.jpg, ... in order
for path in sorted(glob.glob("output/line_*.jpg")):
    img = Image.open(path).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values  # turn image into numbers

    # predict text 
    out = model.generate(pixel_values, max_new_tokens=50, output_scores=True, return_dict_in_generate=True)
    text = processor.batch_decode(out.sequences, skip_special_tokens=True)[0]  # token ids into text

    # how likely each chosen token was 
    scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
    probs = scores[0].exp()  # how confident it was about each token (0 to 1)
    conf = probs.mean().item()  # mean confidence for the whole line
    lowest = probs.min().item()  # confidence of its least confident token

    # raw text -> cleaned text | confidence scores
    # for each raw input, print cleaned output and TrOCR's confidence
    print(path, "->", text, "->", clean(text), f"| conf {conf:.2f} | lowest {lowest:.2f}")