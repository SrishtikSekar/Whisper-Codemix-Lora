from sklearn.model_selection import train_test_split

with open("data/train_all.json","r",encoding="utf-8") as f:
    lines=f.readlines()

train,val=train_test_split(lines,test_size=0.1,random_state=42)

with open("data/train.json","w",encoding="utf-8") as f:
    f.writelines(train)

with open("data/val.json","w",encoding="utf-8") as f:
    f.writelines(val)

print("train/Validation split created")    