import torch 
from datasets import load_dataset,Audio
from transformers import WhisperProcessor,WhisperForConditionalGeneration
from jiwer import wer


dataset=load_dataset("json",data_files="data/test.json")
dataset=dataset.cast_column("audio",Audio(sampling_rate=16000))

processor=WhisperProcessor.from_pretrained(
    "openai/whisper-small",
    language="Tamil",
    task="transcribe"
)

model=WhisperForConditionGeneration.from_pretrained(
    "openai/whisper-small"
).cuda()

preds,refs=[],[]


for sample in dataset["train"]:
    inputs=processor(
        sample["audio"]["array"],
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.cuda()

    with torch.no_grad():
        ids=model.generate(inputs)

    preds.append(
        processor.batch_decode(ids,skip_special_tokens=True)[0]
    )
    refs.append(sample["text"])

base_wer=wer(refs,preds)

with open("baselines/base_wer.txt","w") as f:
    f.write(f"Base Whisper-small WER: {base_wer * 100:.2f}%\n")

print("Base WER: ",base_wer)