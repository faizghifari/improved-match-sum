import os
import json
from time import clock
from types import new_class
from utils import read_jsonl, get_data_path
from collections import defaultdict

BERT_RESULT_PATH="./train_save/bert/result/epoch-2_step-118000_ROUGE-0.347226.pt"
GRU_RESULT_PATH="./train_save/bert-gru/result/epoch-3_step-300000_ROUGE-0.345773.pt"
LSTM_RESULT_PATH="./train_save/bert-lstm/result/epoch-2_step-200000_ROUGE-0.345700.pt"
MEANMAX_RESULT_PATH="./train_save/bert-meanmax/result/epoch-3_step-180000_ROUGE-0.345889.pt"
ROBERTA_RESULT_PATH="./train_save/roberta-2/result/epoch-2_step-280000_ROUGE-0.348038.pt"

def read_summary_result(result_path, data_i):
    ret = []
    with open(os.path.join(result_path,"dec",(str(data_i) + ".dec"))) as sum_file:
        for line in sum_file:
            ret.append(line.strip())
    return ret

def read_summary_reference(result_path, data_i):
    ret = []
    with open(os.path.join(result_path,"ref",(str(data_i) + ".ref"))) as sum_file:
        for line in sum_file:
            ret.append(line.strip())
    return ret

def write_list(lst):
    for lin in lst:
        print(lin.strip())

def interactive_mode():
    mode = "test"
    lang = "en"
    encoder = "bert"
    data_paths = get_data_path(mode, encoder, lang)
    print("Loading data")
    list_data = read_jsonl(data_paths[mode])
    print("Finished loading data")
    n_data= len(list_data) 

    input_i = -1
    while True:
        input_i = input("enter index between 0 - {}: ".format(str(n_data)))
        input_i = int(input_i)
        if input_i == -1:
            break
        if input_i < 0 or input_i > n_data:
            print("invalid index")
            continue
        print(">> TEXT {} =============================================".format(str(input_i)))
        print(write_list(list_data[input_i]["text"]))

        print(">> REF SUMMARY {} =============================================".format(str(input_i)))
        print(write_list(list_data[input_i]["summary"]))

        print(">> BERT SUMMARY {} =============================================".format(str(input_i)))
        write_list(read_summary_result(BERT_RESULT_PATH, input_i))

        print(">> BERT-GRU {} =============================================".format(str(input_i)))
        write_list(read_summary_result(GRU_RESULT_PATH, input_i))

        print(">> BERT-LSTM {} =============================================".format(str(input_i)))
        write_list(read_summary_result(LSTM_RESULT_PATH, input_i))

        print(">> BERT-MEANMAX {} =============================================".format(str(input_i)))
        write_list(read_summary_result(MEANMAX_RESULT_PATH, input_i))

        print(">> ROBERTA {} =============================================".format(str(input_i)))
        write_list(read_summary_result(ROBERTA_RESULT_PATH, input_i))
    print("exit")

# def compare_result(res_1, res_2):
#     ret_bool = True
#     same_sentence = 0
#     if len(res_1) != len(res_2):
#         ret_bool = False
#     for i_1 in res_1:
#         if i_1 not in res_2:
#             ret_bool = False
#         else:
#             same_sentence += 1
#     return ret_bool, same_sentence

def combine_list(*args):
    new_lst = []
    for lst in args:
        new_lst += lst
    return set(new_lst)

def non_interactive_mode():
    list_data = os.listdir(os.path.join(BERT_RESULT_PATH, "dec"))
    total_data = len(list_data)
    sentence_wise = defaultdict(int)
    summary_wise = defaultdict(list)
    order_summary_wise = defaultdict(list)
    for data_i in list_data:
        index_i = int(data_i.split(".")[0])
        bert_result = read_summary_result(BERT_RESULT_PATH, index_i)
        gru_result = read_summary_result(GRU_RESULT_PATH, index_i)
        lstm_result = read_summary_result(LSTM_RESULT_PATH, index_i)
        meanmax_result = read_summary_result(MEANMAX_RESULT_PATH, index_i)
        roberta_result = read_summary_result(ROBERTA_RESULT_PATH, index_i)
        comb_result = combine_list(bert_result, gru_result, lstm_result, meanmax_result, roberta_result)
        
        # no_ord_same = False # TEMP
        # ord_same = False # TEMP

        if len(bert_result) == len(comb_result):
            # all the result is the same
            summary_wise["ALL"] += [index_i]
            sentence_wise["ALL"] += len(comb_result)
        else:
            bert_sent = []
            gru_sent = []
            lstm_sent = []
            meanmax_sent = []
            roberta_sent = []
            for i, sent in enumerate(comb_result):
                key_sent = ""
                if sent in bert_result:
                    bert_sent.append(i)
                    key_sent += "bert-"
                if sent in gru_result:
                    gru_sent.append(i)
                    key_sent += "gru-"
                if sent in lstm_result:
                    lstm_sent.append(i)
                    key_sent += "lstm-"
                if sent in meanmax_result:
                    meanmax_sent.append(i)
                    key_sent += "meanmax-"
                if sent in roberta_result:
                    roberta_sent.append(i)
                    key_sent += "roberta-"
                sentence_wise[key_sent] += 1
            
            # check summary wise            
            bert_different = True
            gru_different = True
            lstm_different = True
            meanmax_different = True
            roberta_different = True
            if bert_sent == gru_sent:
                summary_wise["bert-gru"]+= [index_i]
                bert_different = False
                gru_different = False
            
            if bert_sent == lstm_sent:
                summary_wise["bert-lstm"]+= [index_i]
                bert_different = False
                lstm_different = False
                # no_ord_same = True # TEMP

            if bert_sent == meanmax_sent:
                summary_wise["bert-meanmax"]+= [index_i]
                bert_different = False
                meanmax_different = False

            if bert_sent == roberta_sent:
                summary_wise["bert-roberta"]+= [index_i]
                bert_different = False
                roberta_different = False

            if gru_sent == lstm_sent:
                summary_wise["gru-lstm"]+= [index_i]
                gru_different = False
                lstm_different = False

            if gru_sent == meanmax_sent:
                summary_wise["gru-meanmax"]+= [index_i]
                gru_different = False
                meanmax_different = False

            if gru_sent == roberta_sent:
                summary_wise["gru-roberta"]+= [index_i]
                gru_different = False
                roberta_different = False

            if lstm_sent == meanmax_sent:
                summary_wise["lstm-meanmax"]+= [index_i]
                lstm_different = False
                meanmax_different = False

            if lstm_sent == roberta_sent:
                summary_wise["lstm-roberta"]+= [index_i]
                lstm_different = False
                roberta_different = False

            if meanmax_sent == roberta_sent:
                summary_wise["meanmax-roberta"]+= [index_i]
                meanmax_different = False
                roberta_different = False

            if bert_different:
                summary_wise["bert"] += [index_i]
            if gru_different:
                summary_wise["gru"] += [index_i]
            if lstm_different:
                summary_wise["lstm"] += [index_i]
            if meanmax_different:
                summary_wise["meanmax"] += [index_i]
            if roberta_different:
                summary_wise["roberta"] += [index_i]

        # =====================================================
        o_bert_different = True
        o_gru_different = True
        o_lstm_different = True
        o_meanmax_different = True
        o_roberta_different = True
        if bert_result == gru_result:
            order_summary_wise["bert-gru"] += [index_i]
            o_bert_different = False
            o_gru_different = False
        
        if bert_result == lstm_result:
            order_summary_wise["bert-lstm"] += [index_i]
            o_bert_different = False
            o_lstm_different = False
            # ord_same = True # TEMP

        if bert_result == meanmax_result:
            order_summary_wise["bert-meanmax"] += [index_i]
            o_bert_different = False
            o_meanmax_different = False

        if bert_result == roberta_result:
            order_summary_wise["bert-roberta"] += [index_i]
            o_bert_different = False
            o_roberta_different = False

        if gru_result == lstm_result:
            order_summary_wise["gru-lstm"] += [index_i]
            o_gru_different = False
            o_lstm_different = False
        
        if gru_result == meanmax_result:
            order_summary_wise["gru-meanmax"] += [index_i]
            o_gru_different = False
            o_meanmax_different = False

        if gru_result == roberta_result:
            order_summary_wise["gru-roberta"] += [index_i]
            o_gru_different = False
            o_roberta_different = False

        if lstm_result == meanmax_result:
            order_summary_wise["lstm-meanmax"] += [index_i]
            o_lstm_different = False
            o_meanmax_different = False

        if lstm_result == roberta_result:
            order_summary_wise["lstm-roberta"] += [index_i]
            o_lstm_different = False
            o_roberta_different = False

        if meanmax_result == roberta_result:
            order_summary_wise["meanmax-roberta"] += [index_i]
            o_meanmax_different = False
            o_roberta_different = False

        if o_bert_different:
            order_summary_wise["bert"] += [index_i]
        if o_gru_different:
            order_summary_wise["gru"] += [index_i]
        if o_lstm_different:
            order_summary_wise["lstm"] += [index_i]
        if o_meanmax_different:
            order_summary_wise["meanmax"] += [index_i]
        if o_roberta_different:
            order_summary_wise["roberta"] += [index_i]

        # if no_ord_same and not ord_same: # TEMP
            # print(">>", index_i) # TEMP

        if index_i % 999 == 0:
            print("{},{}".format(index_i, total_data))
        
    # # save the result 
    with open("sentence_wise.json", 'w') as result_file:
        json.dump(sentence_wise, result_file, indent=2)

    with open("summary_wise.json", 'w') as result_file:
        json.dump(summary_wise, result_file, indent=2)

    with open("order_summary_wise.json", 'w') as result_file:
        json.dump(order_summary_wise, result_file, indent=2)

def get_n_summary_wise():
    sw = None
    with open("summary_wise.json", "r") as result_file:
        sw = json.load(result_file)

    old_key = list(sw.keys())
    for key in old_key:
        n_key = "n_" + key
        n_total = len(sw[key])
        sw[n_key] = n_total

    with open("summary_wise.json", 'w') as result_file:
        json.dump(sw, result_file, indent=2)

    o_sw = None
    with open("order_summary_wise.json", "r") as result_file:
        o_sw = json.load(result_file)

    old_key = list(o_sw.keys())
    for key in old_key:
        n_key = "n_" + key
        n_total = len(o_sw[key])
        o_sw[n_key] = n_total

    with open("order_summary_wise.json", 'w') as result_file:
        json.dump(o_sw, result_file, indent=2)

def create_venn_diagram():
    sw = None
    with open("order_summary_wise.json", "r") as result_file:
        sw = json.load(result_file)
    # with open("summary_wise.json", "r") as result_file:
    #     sw = json.load(result_file)

    all_data = list(range(11489))

    old_key = list(sw.keys())
    old_key = [ key for key in old_key if key[:len("n_")] != "n_"]
    print(old_key)
    list_list = [ sw[key] for key in old_key ]
    sum_alldata = combine_list(*list_list)
    print(len(sum_alldata))

    venn_diagram = defaultdict(list)
    for data_i in sum_alldata:
        skip = False
        if data_i in sw["bert-gru"] and \
            data_i in sw["bert-lstm"]and \
            data_i in sw["bert-meanmax"]and \
            data_i in sw["bert-roberta"]:
            venn_diagram["all"] += [data_i]
        # if data_i in sw["ALL"]:
        #     venn_diagram["all"] += [data_i]
            skip = True

        # if skip:
        #     continue
        # 4 group
        if data_i in sw["bert-gru"] and \
            data_i in sw["bert-lstm"]and \
            data_i in sw["bert-meanmax"]:
            venn_diagram["bert-gru-lstm-meanmax"] += [data_i]
            skip = True

        if data_i in sw["bert-gru"] and \
            data_i in sw["bert-lstm"]and \
            data_i in sw["bert-roberta"]:
            venn_diagram["bert-gru-lstm-roberta"] += [data_i]
            skip = True

        if data_i in sw["bert-gru"] and \
            data_i in sw["bert-meanmax"]and \
            data_i in sw["bert-roberta"]:
            venn_diagram["bert-gru-meanmax-roberta"] += [data_i]
            skip = True

        if data_i in sw["bert-lstm"] and \
            data_i in sw["bert-meanmax"]and \
            data_i in sw["bert-roberta"]:
            venn_diagram["bert-lstm-meanmax-roberta"] += [data_i]
            skip = True

        if data_i in sw["gru-lstm"] and \
            data_i in sw["gru-meanmax"]and \
            data_i in sw["gru-roberta"]:
            venn_diagram["gru-lstm-meanmax-roberta"] += [data_i]
            skip = True

        # if skip:
        #     continue
        # 3 group
        if data_i in sw["bert-gru"] and \
            data_i in sw["bert-lstm"]:
            venn_diagram["bert-gru-lstm"] += [data_i]
            skip = True

        if data_i in sw["bert-gru"] and \
            data_i in sw["bert-meanmax"]:
            venn_diagram["bert-gru-meanmax"] += [data_i]
            skip = True

        if data_i in sw["bert-gru"] and \
            data_i in sw["bert-roberta"]:
            venn_diagram["bert-gru-roberta"] += [data_i]
            skip = True

        if data_i in sw["bert-lstm"] and \
            data_i in sw["bert-meanmax"]:
            venn_diagram["bert-lstm-meanmax"] += [data_i]
            skip = True

        if data_i in sw["bert-lstm"] and \
            data_i in sw["bert-roberta"]:
            venn_diagram["bert-lstm-roberta"] += [data_i]
            skip = True

        if data_i in sw["bert-meanmax"] and \
            data_i in sw["bert-roberta"]:
            venn_diagram["bert-meanmax-roberta"] += [data_i]
            skip = True

        if data_i in sw["gru-lstm"] and \
            data_i in sw["gru-meanmax"]:
            venn_diagram["gru-lstm-meanmax"] += [data_i]
            skip = True

        if data_i in sw["gru-lstm"] and \
            data_i in sw["gru-roberta"]:
            venn_diagram["gru-lstm-roberta"] += [data_i]
            skip = True

        if data_i in sw["gru-meanmax"] and \
            data_i in sw["gru-roberta"]:
            venn_diagram["gru-meanmax-roberta"] += [data_i]
            skip = True

        if data_i in sw["lstm-meanmax"] and \
            data_i in sw["lstm-roberta"]:
            venn_diagram["lstm-meanmax-roberta"] += [data_i]
            skip = True

        # if skip:
        #     continue
        # 2 group
        if data_i in sw["bert-gru"]:
            venn_diagram["bert-gru"] += [data_i]
            skip = True

        if data_i in sw["bert-lstm"]:
            venn_diagram["bert-lstm"] += [data_i]
            skip = True

        if data_i in sw["bert-meanmax"]:
            venn_diagram["bert-meanmax"] += [data_i]
            skip = True

        if data_i in sw["bert-roberta"]:
            venn_diagram["bert-roberta"] += [data_i]
            skip = True

        if data_i in sw["gru-lstm"]:
            venn_diagram["gru-lstm"] += [data_i]
            skip = True

        if data_i in sw["gru-meanmax"]:
            venn_diagram["gru-meanmax"] += [data_i]
            skip = True

        if data_i in sw["gru-roberta"]:
            venn_diagram["gru-roberta"] += [data_i]
            skip = True

        if data_i in sw["lstm-meanmax"]:
            venn_diagram["lstm-meanmax"] += [data_i]
            skip = True

        if data_i in sw["lstm-roberta"]:
            venn_diagram["lstm-roberta"] += [data_i]
            skip = True

        if data_i in sw["meanmax-roberta"]:
            venn_diagram["meanmax-roberta"] += [data_i]
            skip = True

        # 1 group
        # if skip:
        #     continue

        if data_i in sw["bert"]:
            venn_diagram["bert"] += [data_i]

        if data_i in sw["gru"]:
            venn_diagram["gru"] += [data_i]

        if data_i in sw["lstm"]:
            venn_diagram["lstm"] += [data_i]

        if data_i in sw["meanmax"]:
            venn_diagram["meanmax"] += [data_i]

        if data_i in sw["roberta"]:
            venn_diagram["roberta"] += [data_i]

    # print(json.dumps(venn_diagram, indent=2))
    with open("venn_diagram_order.json", 'w') as venfila:
        json.dump(venn_diagram, venfila, indent=2)
    # with open("venn_diagram_no_order.json", 'w') as venfila:
    #     json.dump(venn_diagram, venfila, indent=2)

if __name__ == "__main__":
    # interactive_mode()
    # non_interactive_mode()
    # get_n_summary_wise()  
    # create_venn_diagram()
    vd = None
    # with open("venn_diagram_no_order.json", 'r') as venfila:
    with open("venn_diagram_order.json", 'r') as venfila:
        # json.dump(venn_diagram, venfila, indent=2)
        vd = json.load(venfila)

    vd_key = list(vd.keys())
    # list_list = [ vd[key] for key in vd_key ]
    # sum_alldata = combine_list(*list_list)
    # print(len(sum_alldata))
    vd_key = [ key.split("-") for key in vd_key ]

    list_model = ["bert","gru", "lstm", "meanmax", "roberta"]
    for keym in list_model:
        total = len(vd["all"])
        print(keym, "all", total)
        for vkey in vd_key:
            if keym in vkey:
                n_data = len(vd["-".join(vkey)])
                print(keym, vkey, n_data)
                total += n_data
        print(keym, ":", total)

