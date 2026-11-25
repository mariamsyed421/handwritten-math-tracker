
# this program just reads an example single line of math  with TrOCR 
# this is just an example for testing and not needed for our actual project

from transformers import TrOCRProcessor, VisionEncoderDecoderModel #hugging Face's library for loading pretrained models
from PIL import Image #open image files

# prepares images going in, turns output back into text
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# trained model that reads the image
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

img = Image.open("line1_clean.jpg").convert("RGB")  # TrOCR expects color images, convert to RGB 
pixel_values = processor(images=img, return_tensors="pt").pixel_values  # resize + turn image into numbers
ids = model.generate(pixel_values, max_new_tokens=50)  # predict token ids
print(processor.batch_decode(ids, skip_special_tokens=True)[0])  # token ids -> text

c