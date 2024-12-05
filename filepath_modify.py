import json

input_filepath = "/home/aix23606/seoah/JejuDialectSTT/data_sample/dataset.json"
outputh_filepath = "/home/aix23606/seoah/JejuDialectSTT/data_sample/dataset.json"

original_path = "/home/aix23606/seoah/JejuDialectSTT/data_sample/Trimmed_data_2인발화/Trimmed_data"
replacement_path = "/home/aix23606/seoah/JejuDialectSTT/data_sample/Trimmed_data_2인발화"

with open(input_filepath, "r", encoding="utf-8") as file:
    data = json.load(file)

for item in data["dataset"]:
    item["audio_filepath"] = item["audio_filepath"].replace(original_path, replacement_path)

with open(outputh_filepath, "w", encoding="utf-8") as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

print("Done!")

