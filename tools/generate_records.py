import pandas as pd
import math
def main():
    exams = pd.read_csv("../balanced_5000/balanced_5000.csv")
    counter = 1
    for exam_ids in exams['exam_id']:
        #print(exam_ids)
        exam_ids = str(exam_ids)
        filled_id = exam_ids.zfill(7)
        S_val = 'S' + filled_id[0:3].ljust(7,'0')
        #print(S_val)
        four_dig_val = filled_id[0:4].zfill(4)
        #print(four_dig_val)
        #print(four_dig_val)
        last_val = str(exam_ids).zfill(7)
        tnmg_val = "TNMG" + exam_ids + "_N1"
        full_path = "/data/isip/data/tnmg_code/v1.0.0/data/" + S_val + "/" + four_dig_val +  "/" + last_val + "/" + tnmg_val
        
        #print(full_path)
        with open("../balanced_5000/balanced_5000_records.txt", 'a') as records:
            records.write(full_path + "\n") 
            records.close()
        counter = counter + 1
if __name__ == "__main__":
    main()
