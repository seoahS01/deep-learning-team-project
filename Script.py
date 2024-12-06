import os
import json
from pydub import AudioSegment


def cut_audio(file_path, start_time, end_time, output_path):
    """
    오디오 파일에서 특정 구간을 잘라내서 저장
    """
    # 시간 문자열을 밀리초로 변환
    def time_to_milliseconds(time_str):
        h, m, s = time_str.split(":")
        s, ms = map(float, s.split("."))
        return int((int(h) * 3600 + int(m) * 60 + s) * 1000 + ms)
    
    #오디오가 짤려서 살짝 조정
    start_ms = time_to_milliseconds(start_time) - 50
    end_ms = time_to_milliseconds(end_time) + 100

    # 오디오 파일 로드
    audio = AudioSegment.from_file(file_path)

    # 특정 구간 잘라내기
    trimmed_audio = audio[start_ms:end_ms]

    # 잘라낸 오디오 저장
    trimmed_audio.export(output_path, format="wav")
    print(f"오디오 저장 경로: {output_path}")

def save_dict_to_json(dictionary, file_path):
    """
    딕셔너리를 JSON 파일로 저장.
    """
    # JSON 파일로 저장
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(dictionary, json_file, ensure_ascii=False, indent=4)

#전처리 결과를 저장할 딕셔너리
input_data = {'dataset':[]}

# JSON 파일들이 들어있는 폴더 경로
folder_path = r"C:/Users/김우영/Desktop/2024-2/딥러닝/project/Sample/02.라벨링데이터/03. 제주도/01. 1인발화 따라말하기"

# 폴더 내 모든 파일에 대해 반복
for filename in os.listdir(folder_path):
    # 파일이 JSON 확장자인지 확인
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"{filename} 내용: {data['script']['domain']}")

            script_list = data['transcription']['segments']

            for word in script_list:
                #파일명
                file_name = data['fileName'] + 'order_' + str(word['orderInFile']) + '.wav'
                #파일 저장 경로
                save_file_path = r'C:/Users/김우영\Desktop/2024-2/딥러닝/project/Trimmed_data/' + file_name
                start_time = word['startTime']
                end_time = word['endTime']
                #스크립트
                text = word['dialect']
                #파일경로, 텍스트 저장
                data_list = input_data['dataset']
                current_data = {'audio_filepath' : save_file_path, 'text' : text}
                data_list.append(current_data)
                input_data['dataset'] = data_list
                #오디오 파일 단어별로 잘라서 저장
                load_file_path = str(file_path)
                load_file_path = load_file_path.replace('02.라벨링데이터','01.원천데이터')
                load_file_path = load_file_path.replace('.json','.wav')

                print(load_file_path)
                cut_audio(load_file_path, start_time, end_time, save_file_path)

#결과 json파일로 저장
save_dict_to_json(input_data, r'C:/Users/김우영/Desktop/2024-2/딥러닝/project/dataset.json')

