import json
import openai
import random
import os

prefix = os.getcwd()
# names_json = os.path.join(prefix, "names.json")
goal_mapping_json = os.path.join(prefix, "data/goal_mapping.json")
goal_set_json = os.path.join(prefix, "data/groundtruth.json")
task_info_json = os.path.join(prefix, "data/task_info.json")
goal_lib_json = os.path.join(prefix, "data/goal_lib.json")
task_prompt_file = os.path.join(prefix, "data/task_prompt.txt")
parse_prompt_file = os.path.join(prefix, "data/parse_prompt.txt")
openai_keys_file = os.path.join(prefix, "data/openai_keys.txt")

class Planner:
    def __init__(self):
        self.task_prompt = self.load_prompt(task_prompt_file)
        self.parser_prompt = self.load_prompt(parse_prompt_file)
        # print(self.parser_prompt)
        self.dialogue = ''
        self.goal_lib = self.load_goal_lib()
        self.openai_api_keys = self.load_openai_keys()
        self.supported_objects = self.get_supported_objects(self.goal_lib)

    def load_goal_lib(self, ):
        with open(goal_lib_json,'r') as f:
            goal_lib = json.load(f)
        return goal_lib
    
    def load_openai_keys(self,):
        with open(openai_keys_file, "r") as f:
            context = f.read()
        print(context.split('\n'))
        return context.split('\n')
    
    def get_supported_objects(self, goal_lib):
        supported_objs = {}
        for key in goal_lib.keys():
            obj = list(goal_lib[key]['output'].keys())[0]
            supported_objs[obj] = goal_lib[key]
            supported_objs[obj]['name'] = key
        return supported_objs

    def load_prompt(self, file):
        with open(file, 'r') as f:
            context = f.read()
        return context
    
    def query_codex(self, prompt_text):
        server_flag = 0
        while True:
            try:
                self.update_key()
                response =  openai.Completion.create(
                    model="code-davinci-002",
                    prompt=prompt_text,
                    temperature=0.7,
                    max_tokens=1024,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=["Human:",]
                    )
                # result_text = response['choices'][0]['text']
                server_flag = 1
                if server_flag:
                    break
            except Exception as e:
                print(e)
        return response
        

    def query_gpt3(self, prompt_text):
        server_flag = 0
        while True:
            try:
                self.update_key()
                response =  openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompt_text,
                    temperature=0,
                    max_tokens=256,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                result_text = response['choices'][0]['text']
                # print("parse prompt:", prompt_text)
                # print("parse_result:", result_text)
                server_flag = 1
                if server_flag:
                    break
            except Exception as e:
                print(e)
        return response

    def update_key(self, ):
        # openai_api_keys = self.openai_keys
        curr_key = self.openai_api_keys[0]
        openai.api_key = curr_key
        self.openai_api_keys.remove(curr_key)
        self.openai_api_keys.append(curr_key)

    def online_parser(self, text):
        parser_prompt_text = self.parser_prompt + text
        response = self.query_gpt3(parser_prompt_text)
        parsed_info = response["choices"][0]["text"]
        print(parsed_info)
        lines = parsed_info.split('\n')

        name = None
        obj = None
        rank = None
        
        for line in lines:
            line = line.replace(' ', '')
            # print('line', line)
            if 'action:' in line:
                action = line[7:]
                # print(f"action: {action}")
            elif 'name:' in line:
                name = line[5:]
                # print(f"name: {name}")
            elif 'object:' in line:
                obj = eval(line[7:])
                # print("[INFO]:", type(obj))
                # print(f"object: {obj}")
            elif 'rank:' in line:
                rank = int(line[5:])
                # print(f"rank: {rank}")
            else:
                # print("Error parsing the goal.")
                pass
        return name, obj, rank

    def check_object(self, object):
        object_flag = False
        try:
            object_name = list(object.keys())[0]
            for goal in self.goal_lib.keys():
                if object_name == list(self.goal_lib[goal]["output"].keys())[0]:
                    object_flag = True
                    return goal
        except Exception as e:
            print(e)
        return object_flag


    def generate_goal_list(self, plan):
        lines = plan.split('\n')
        goal_list = []
        for line in lines:
            if '#' in line:
                name, obj, rank = self.online_parser(f"input: {line}")
                print("[INFO]: ", name, obj, rank)

                if name in self.goal_lib.keys():
                    goal_type = self.goal_lib[name]["type"]
                    goal_object = obj
                    goal_rank = rank
                    goal_precondition = {**self.goal_lib[name]["precondition"], **self.goal_lib[name]["tool"]}
                    goal = {}
                    goal["name"] = name
                    goal["type"] = goal_type
                    goal["object"] = goal_object
                    goal["precondition"] = goal_precondition
                    goal["ranking"] = goal_rank
                    goal_list.append(goal)
                elif self.check_object(obj):
                    print("[INFO]: parsed goal is not in controller goal keys. Now search the object items ...")
                    obj_name = list(obj.keys())[0]
                    goal_type = self.supported_objects[obj_name]["type"]
                    goal_object = obj
                    goal_rank = rank
                    goal_precondition = {**self.supported_objects[obj_name]["precondition"], **self.supported_objects[obj_name]["tool"]}
                    goal = {}
                    goal["name"] = self.supported_objects[obj_name]["name"]
                    goal["type"] = goal_type
                    goal["object"] = goal_object
                    goal["precondition"] = goal_precondition
                    goal["ranking"] = goal_rank
                    goal_list.append(goal)
                else:
                    print("[ERROR]: parsed goal is not supported by current controller.")
        print(f"[INFO]: Current Plan is {goal_list}")
        return goal_list

    def initial_planning(self, task_question):
        question = f"Human: {task_question}\n"
        task_prompt_text = self.task_prompt + question
        response = self.query_codex(task_prompt_text)
        plan = response["choices"][0]["text"]
        self.dialogue = task_prompt_text
        self.dialogue += plan
        print(plan)
        return plan

    def generate_inventory_description(self, inventory):
        inventory_text = 'Human: My inventory now has '
        for inv_item in inventory:
            if not inv_item['name'] == 'air':
                inventory_text += f'{inv_item["quantity"]} {inv_item["name"]},'
        print(inventory_text)
        inventory_text += '\n'
        self.dialogue += inventory_text
        return inventory_text
        

    def generate_success_description(self, step):
        result_description = f'Human: I succeed on step {step}.\n'
        self.dialogue += result_description
        return result_description

    def generate_failure_description(self, step):
        result_description = f'Human: I fail on step {step}'
        self.dialogue += result_description
        print(result_description)
        return result_description


    def generate_explanation(self):
        response = self.query_codex(self.dialogue)
        explanation = response["choices"][0]["text"]
        self.dialogue += explanation
        print(explanation)
        return explanation

    def replan(self, task_question):
        replan_description = f"Human: Please fix above errors and replan the task '{task_question}'.\n"
        self.dialogue += replan_description
        response = self.query_codex(self.dialogue)
        plan = response["choices"][0]["text"]
        print(plan)
        self.dialogue += plan
        return plan

    