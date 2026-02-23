import os
import json


class Logger:
    def __init__(self, log_path, remove_existing=True):
        os.makedirs(log_path, exist_ok=True)
        if remove_existing:
            for file in os.listdir(log_path):
                file_path = os.path.join(log_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        self.train_log_stream = open(os.path.join(log_path, 'train.jsonl'), 'a')
        self.eval_log_stream = open(os.path.join(log_path, 'eval.jsonl'), 'a')

    def __del__(self):
        self.train_log_stream.close()
        self.eval_log_stream.close()

    def log(self, message: dict,is_train=True):
        if is_train:
            self.train_log_stream.write(json.dumps(message) + '\n')
            self.train_log_stream.flush()
        else:
            self.eval_log_stream.write(json.dumps(message) + '\n')
            self.eval_log_stream.flush()