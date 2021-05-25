from PIL import ImageFont,ImageDraw,Image
import numpy as np
import itertools
from urllib.parse import unquote
import pandas as pd

class module:
    # ====================================================================
    # __init__
    if True:
        # --------------------------------------------------------------------
        # 주 메서드
        def __init__(self):
            # 폰트 설정 ** 갓터리 - 파란망고 - 타이포럭키 순서를 지켜주세요 ** 
            font_path = 'C:/Users/master11/Desktop/OUTPUT_WEB(ver 1.1.1)/font/'
            self.FONT=[font_path+'OdGodttery.ttf',font_path+'LAB파란망고.ttf',
            font_path+'Typo_luckypangB.ttf']

    # ====================================================================
    # letter_to_image
    if True:
        # --------------------------------------------------------------------
        # 주 메서드
        def letter_to_image(self,letter_list):
            input_image = []
            for ttf in self.FONT:
                ttf_image = []
                for letter in letter_list:
                  ttf_image.append(self.txt_to_bmp(letter, ttf))
                input_image.append(ttf_image)
            return input_image

        # --------------------------------------------------------------------
        # 보조 메서드
        if True:    
            def txt_to_bmp(self,text,use_font):
                chars = text
                font = ImageFont.truetype(font=use_font, size=100)
                x, y = font.getsize(chars)

                img_ground = Image.new('L', (x+3, y+3), color='white')
                img_draw = ImageDraw.Draw(img_ground)
                img_draw.text((0, 0), chars, font=font, fill='black')

                img = img_ground.resize((100, 100))
                img = np.array(img)
                img = abs(img/255-1)
                return img 

    # ====================================================================
    # sent_case
    if True:
        # --------------------------------------------------------------------
        # 주 메서드
        def sent_case(self,groups):
            result = []
            if len(groups) == 2:
                return list(itertools.product(groups[0], groups[1], repeat=1))
            elif len(groups) == 1:
                return list(itertools.product(groups[0], repeat=1))
            else:
                roots = groups[0]
                tmp = []
                for root in roots:
                    tmp = (root,)
                    for i in self.sent_case(groups[1:]):
                        tmp += i
                        result.append(tmp)
                        tmp = (root,)
            return result


    # ====================================================================
    # change_sentence
    if True:
        # --------------------------------------------------------------------
        # 주 메서드
        def change_sentence(self,text, id_list, input_df):
            text = text.replace(' ', '')
            ID = [i for i in range(len(id_list))]
            for i, j in zip(ID, id_list):
                tmp = input_df[(input_df['result_id'] == i) &
                                (input_df['partition_id'] == j)]
                # 바꿀 대상
                for a, pred in zip(tmp['letter'].values, tmp['pred_letter_best'].values):
                    text = text.replace(a, pred, 1)
            return text


    # ====================================================================
    # filter_msg
    if True:
        # --------------------------------------------------------------------
        # 주 메서드
        def filter_msg(self,data,filtering_keyword):
            username=data['username']
            msg=data['msg']
            text=unquote(msg,encoding='utf-8')

            filter_need=False
            filter_why = ''

            for key in filtering_keyword:
                if key in text:
                    filter_need=True
                    filter_why = key
                    break
            
            return filter_need,username,msg,text,filter_why

    # ====================================================================
    # input_preprocess
    if True:
        # --------------------------------------------------------------------
        # 주 메서드
        def input_preprocess(self,input_data):
            Id = [] 
            Part_Id = []
            letter = []

            for key in input_data.keys():
                part_id = 0
                for partition in input_data[key]:
                    Id.extend([key for _ in range(len(partition))])
                    Part_Id.extend([part_id for _ in range(len(partition))])
                    letter.extend([letter for letter in partition])
                    part_id += 1

            input_dict = {'result_id': Id,
                        'partition_id': Part_Id,
                        'letter': letter}
            input_df = pd.DataFrame(input_dict)
            input_df['pred_letter_440'] = [np.nan for _ in range(len(input_df))] # 나중에 예측결과가 올 것
            input_df['pred_letter_385'] = [np.nan for _ in range(len(input_df))] # 나중에 예측결과가 올 것
            input_df['pred_letter_759'] = [np.nan for _ in range(len(input_df))] # 나중에 예측결과가 올 것

            input_df['pred_score_440']=[np.nan for _ in range(len(input_df))] # 나중에 예측확률이 올 것
            input_df['pred_score_385'] = [np.nan for _ in range(len(input_df))] # 나중에 예측확률이 올 것
            input_df['pred_score_759'] = [np.nan for _ in range(len(input_df))] # 나중에 예측확률이 올 것

            input_df['pred_letter_best'] = [np.nan for _ in range(len(input_df))] # 나중에 최종예측결과가 올 것
            
            return input_df

    # ====================================================================
    # predict_result
    if True:
        # --------------------------------------------------------------------
        # 주 메서드
        def predict_result(self,model,input_imgs,input_df,letter_df):
            result_ind = letter_df['letter'].values
            pred = model.predict(input_imgs)
            result_letters = result_ind[np.argmax(pred, axis=1)]
            result_letters = result_letters.reshape(3,len(result_letters)//3)
            result_scores = np.array(list((map(max,pred))))
            result_scores = result_scores.reshape(3,len(result_scores)//3)

            input_df['pred_letter_440'] = result_letters[0]
            input_df['pred_letter_385'] = result_letters[1]
            input_df['pred_letter_759'] = result_letters[2]

            # 가중치 불러오기 
            # [440: 0.34459755, 385: 0.33555955, 759: 0.31984288]
            w1,w2,w3 = np.load('files/weights_random.npy')

            input_df['pred_score_440'] = result_scores[0] * w1
            input_df['pred_score_385'] = result_scores[1] * w2
            input_df['pred_score_759'] = result_scores[2] * w3

            for i in range(len(input_df)):
                pred_letters_array = input_df.iloc[i,3:6].values
                pred_letters_scores =input_df.iloc[i,6:9].values
            
                # 가중치 부여 후 제일 높은 걸로
                input_df.iloc[i,-1] = pred_letters_array[np.argmax(pred_letters_scores)]

    # ====================================================================
    # comp_to_len
    if True:
        # --------------------------------------------------------------------
        # 주 메서드
        def comp_to_len(self,input_data):
            for key in input_data:
                input_data[key] = list(range(len(input_data[key])))




def filter_msg(data,filtering_keyword):
    username=data['username']
    msg=data['msg']
    text=unquote(msg,encoding='utf-8')

    filter_need=False

    for key in filtering_keyword:
        if key in text:
            filter_need=True
    
    return filter_need,username,msg,text