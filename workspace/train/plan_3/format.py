import csv
import json

from workspace.env import workspace_in_win

if __name__ == '__main__':
    csv_file=f'{workspace_in_win}\\train\\plan_3\\input\\ahc客服话术.csv'
    with open(csv_file,mode='r',encoding='utf-8') as f:
        with open(f'{workspace_in_win}/train/plan_3/input/train.json', mode='w', encoding='utf-8') as wf:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                if len(line)<3:
                    continue
                x=json.dumps({
                    'question':line[1],
                    'answer':line[2]
                })
                wf.write(x+"\n")

