from flask import Flask, render_template,request,jsonify
from flask_socketio import SocketIO, emit
import tensorflow as tf
from werkzeug.utils import redirect
from text_Preprocess import preprocessor
import pandas as pd
import numpy as np
from api import module
import jpype
# GPU 설정하기
import os

# GPU 설정
if True:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 인스턴스 호출
if True:
    mod=module()
    processor=preprocessor()

# 모델 호출
if True:
    model_path = os.path.join('Korean_classifier','all_letters_jds_acc_96')
    model=tf.keras.models.load_model(model_path)

# 예측값에 해당하는 인덱스 목록
letter_df = pd.read_csv(os.path.join('files','csv','Labels.csv'))

# app 호출
if True:
    app = Flask(__name__)
    app.jinja_env.add_extension('jinja2.ext.loopcontrols')
    app.config['SECRET_KEY'] = 'miniproject'

# 금지어 설정
filtering_keyword =['사과티비']


# socketio 통신 관련
if True:
    socketio = SocketIO(app, cors_allowed_origins='*', pingInterval=600000,
                        pingTimeout=600000, async_mode='threading', manage_session=False)
    
    
    @socketio.on('connect')
    def connect():
        print('유저 접속')
        msg = '모델 테스트용 채팅 서비스입니다.'
        emit('s_send_msg', {'user': 'System', 'msg': msg}, broadcast=True)

    
    @socketio.on('c_send_msg')
    def c_send_msg(data):  # 요기서 댓글 텍스트 처리
        
        # 스레드 관리
        jpype.attachThreadToJVM()

        # 데이터 받아서 금지어 있으면 pass, 없으면 분석
        filter_need,username,msg,text,filter_why=mod.filter_msg(data,filtering_keyword)

        # output은 변화하기 전으로 출력하기 위해서 text를 한번 더 저장합니다
        output_text = text
        
        while True:
            if filter_need:
                break
            input_data = processor(text)
            # 최종 검사텍스트에서는 변환된 상태에서 체크하기때문에 변환상태로 만듭니다
            text = processor.complete_fake_to_real(text)

            # 정상적인 문구인데, 띄어쓰기가 되어 있는 경우 like 사과 티비
            if not input_data:
                print('정상적인 문구를 조사합니다')
                tmp_text = text
                tmp_text = tmp_text.replace(' ', '')
                for keyword in filtering_keyword:
                    if keyword in tmp_text:
                      filter_need = True
                      filter_why = keyword
                      break
                break

            input_df=mod.input_preprocess(input_data)

            input_imgs = np.array(mod.letter_to_image(input_df['letter'].values))
            input_imgs = input_imgs.reshape(input_imgs.shape[1]*3,100,100,1)
            mod.predict_result(model,input_imgs,input_df,letter_df)

            result_text = text.replace(' ','')
            mod.comp_to_len(input_data)
            case = [input_data[key]for key in input_data]
            
            # 데카르트 곱연산 결과
            result_case = mod.sent_case(case)
            # 최종문장후보
            result_text_list = list()

            for id_list in result_case:
                result_text_list.append(mod.change_sentence(text,id_list,input_df))


            for result_text in result_text_list:
                if filter_need:
                    break   
                for keyword in filtering_keyword:
                    if keyword in result_text:
                        filter_need=True
                        filter_why = keyword
                        break
            
            break


        if filter_need:
            msg=f'[필터링 되었습니다] 이유: {filter_why}'
            emit('s_send_msg', {'user': 'test', 'msg': msg}, broadcast=True) 
        else:
            emit('s_send_msg', {'user': 'test', 'msg': output_text}, broadcast=True)

    
    # 9. 유저 이탈

    @socketio.on('disconnect')
    def disconnect():
        print('유저 이탈')


@app.route('/')
def chat():
    return render_template('index2.html', keywords = filtering_keyword)

@app.route('/ban',methods = ['GET','POST'])
def ban():
    if request.method=='POST':
        ban = request.form['ban_id']
        if ban in filtering_keyword:
            return jsonify('no')
        else:
            filtering_keyword.append(ban)
            return jsonify(ban)
    else:
        pass


if __name__ == '__main__':
    socketio.run(app, debug=True)
