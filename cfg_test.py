import yaml

with open("./config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

# for section in cfg:
#     print(section)
print(cfg["openai"])
print(cfg["es"])