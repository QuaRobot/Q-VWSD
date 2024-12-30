from openai import OpenAI
import json
from tqdm import tqdm
import threading

res ={}
with open('gpt3_5_train.json', 'r') as f:
    res = json.load(f)
print(len(res))
tar = []
with open(r"train.data.v1.txt", "r", encoding='utf-8') as f:
    for i in f:
        tar.append(i.split('\t')[0])
print(len(tar))
print(len(set(tar)))

num = 0
for i in tar:
    if i not in res.keys():
        print(i)
        num+=1
print(num)

client = OpenAI(
    base_url="",
    api_key=""
)

num = len(tar)


class MyThread(threading.Thread):

    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result  
        except Exception as e:
            print(e)
            return {}




def task(text):
    dictionary ={}

    if text not in res.keys():
        flag = 0
        while True:

            try:
                    completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an excellent dictionary. Please provide at least five concise and precise interpretations of a given word, focusing on different semantic as much as possible.\n "
                                                  +"Input format: word\n "
                                                  +"Output format: {\"word\":[[gloss1],[glosss2],[gloss3],[gloss4],[gloss5]]}\n "
                                                  +"All content translated into English."},
                        {"role": "user", "content": text}
                    ]
                    )
                    string = completion.choices[0].message.content
                    dictionary = eval(string)
                    break
            except Exception as e:
                flag += 1
                print(e)
                if flag==10:
                    break
    return dictionary



for i in tqdm(range(0,num,10)):
    threads = []
    for j in range(10):
        thread = MyThread(task, (tar[i+j],))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
    for thread in threads:
        res.update(thread.get_result())

    with open('gpt3_5_train.json', 'w') as f:
        json.dump(res, f)

with open('gpt3_5_train.json', 'w') as f:
    json.dump(res, f)


