from pathlib import Path

data_dir = Path(str(Path.cwd())+"/dataset")
train_path = data_dir / 'train.json'
dev_path = data_dir / 'dev.json'
test_path = data_dir / 'test.json'
output_dir = Path(str(Path.cwd())+"/outputs")


label2id = {
    "O": 0,
    "B-PERSON": 1,
    "B-AIRPORT": 2,
    "B-HOSPITAL": 3,
    "B-HOTEL": 4,
    "B-RESTAURANT": 5,
    "B-SPORTS_FIELD":6,
    "B-THEATER": 7,
    "B-RAILWAY": 8,
    "B-ROAD": 9,
    "B-RESIDENCE_COMMUNITY": 10,
    "B-CORPORATION":11,
    "B-GOVERNMENT": 12,
    "B-EDUCATIONAL_INSTITUTIONS": 13,
    "B-LAW_ENFORCEMENT": 14,
    "B-SPORTS_TEAMS": 15,
    "B-OTHERS": 16,
    "B-OFFICE_BUILDING": 17,
    "I-PERSON": 18,
    "I-AIRPORT": 19,
    "I-HOSPITAL": 20,
    "I-HOTEL": 21,
    "I-RESTAURANT": 22,
    "I-SPORTS_FIELD": 23,
    "I-THEATER": 24,
    "I-RAILWAY": 25,
    "I-ROAD": 26,
    "I-RESIDENCE_COMMUNITY": 27,
    "I-CORPORATION": 28,
    "I-GOVERNMENT": 29,
    "I-EDUCATIONAL_INSTITUTIONS": 30,
    "I-LAW_ENFORCEMENT": 31,
    "I-SPORTS_TEAMS": 32,
    "I-OTHERS": 33,
    "I-OFFICE_BUILDING": 34,
    "<START>": 35,
    "<STOP>": 36
}

if __name__ == "__main__":
    for i, label in enumerate(label2id):
        print(i,' ',label)



# "O": 0,
# "B-address":1,
# "B-book":2,
# "B-company":3,
# 'B-game':4,
# 'B-government':5,
# 'B-movie':6,
# 'B-name':7,
# 'B-organization':8,
# 'B-position':9,
# 'B-scene':10,
# "I-address":11,
# "I-book":12,
# "I-company":13,
# 'I-game':14,
# 'I-government':15,
# 'I-movie':16,
# 'I-name':17,
# 'I-organization':18,
# 'I-position':19,
# 'I-scene':20,
# "S-address":21,
# "S-book":22,
# "S-company":23,
# 'S-game':24,
# 'S-government':25,
# 'S-movie':26,
# 'S-name':27,
# 'S-organization':28,
# 'S-position':29,
# 'S-scene':30,
# "<START>": 31,
# "<STOP>": 32