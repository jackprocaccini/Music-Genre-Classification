import os
import csv
import subprocess
import shutil

DIRECTORY_PATH = os.getcwd()
YOUTUBE_DL_PATH = DIRECTORY_PATH + "\\youtube-dl.exe"
FFMPEG_PATH = DIRECTORY_PATH + "\\ffmpeg.exe"
DATA_SET_PATH = ""
YOUTUBE_URL = "https://www.youtube.com/watch?v="

# genres to be downloaded/extracted. Taken from ontology.json
# GENRES_DICTIONARY = {
#     "blues": "/m/06j6l",
#     "classical": "/m/0ggq0m",
#     "country": "/m/01lyv",
#     "disco": "/m/026z9",
#     "hiphop": "/m/0glt670",
#     "jazz": "/m/03_d0",
#     "metal": "/m/03lty",
#     "pop": "/m/064t9",
#     "reggae": "/m/06cqb",
#     "rock": "/m/06by7"
#     }

GENRES_DICTIONARY = {
    "blues": ["/m/06j6l"],
    "classical": ["/m/0ggq0m", "/m/05lls"],
    "country": ["/m/01lyv", "/m/015y_n", "/m/0gg8l"],
    "disco": ["/m/026z9"],
    "hiphop": ["/m/0glt670", "/m/04j_h4", "/m/0n8zsc8", "/m/02cz_7"],
    "jazz": ["/m/03_d0"],
    "metal": ["/m/03lty"],
    "pop": ["/m/064t9"],
    "raggae": ["/m/06cqb", "/m/0190y4"],
    "rock": ["/m/06by7", "/m/05r6t", "/m/0dls3", "/m/0dl5d", "/m/07sbbz2", "/m/05w3f"]
}

def get_genre_audio_info(genre, audio_info):

    list_of_info = []
    
    for row in audio_info:
        for tag in GENRES_DICTIONARY[genre]:
            if tag in row[3]:
                list_of_info.append(row)

    return list_of_info

def download_audio(youtube_id, genre, start=0.000, end=1.000, count=0):
    dir_path = os.getcwd()
    dir_components = dir_path.split("\\")
    output_string = genre + str(count) + ".wav"
    # print("output string for this loop: ", output_string)
    # output_string = "test" + str(count) + ".wav"

    subprocess.call(["youtube-dl", "-x", "--audio-format", "wav", "-o", "%(id)s.%(ext)s", YOUTUBE_URL + youtube_id])
    subprocess.call(["ffmpeg", "-hide_banner", "-loglevel", "warning", "-i", youtube_id + ".wav", "-ss", str(start), "-t", str(end-start), "-acodec", "copy", output_string])
    os.remove(youtube_id + ".wav")

    # exit_code = 0
    # try:
    #     subprocess.check_call(["youtube-dl", "-x", "--audio-format", "wav", "-o", "%(id)s.%(ext)s", YOUTUBE_URL + youtube_id])
    # except subprocess.CalledProcessError as e:
    #     print("Could not download video with id: ", youtube_id)
    #     exit_code = e.returncode

    # if exit_code == 0:
    #     try:
    #        subprocess.call(["ffmpeg", "-hide_banner", "-loglevel", "warning", "-i", youtube_id + ".wav", "-ss", str(start), "-t", str(end-start), "-acodec", "copy", output_string])
    #        os.remove(youtube_id + ".wav")
    #     except Exception as e:
    #         print("Could not trim audio with id: ", youtube_id)

def main():
    # create main data set directory
    os.mkdir(DIRECTORY_PATH + "\\audio_set_data")
    DATA_SET_PATH = DIRECTORY_PATH + "\\audio_set_data"

    # create genre directories
    os.chdir(DATA_SET_PATH)
    print("creating genre directories")
    for keys in GENRES_DICTIONARY:
        os.mkdir(keys)
    
    os.chdir(DIRECTORY_PATH) # brings you back to where the program was run

    # load information from csv file
    with open("eval_segments.csv") as csv_file:
        reader = csv.reader(csv_file, skipinitialspace=True)
        audio_info = list(reader)

    all_audio_info = audio_info[3:] # skips the first 3 lines in eval_segments.csv, since they're comments

    # download youtube vidoes as .wav in correct folder
    # print("current directory before loop: ", os.getcwd())

    for genre in GENRES_DICTIONARY.keys():
        print("Current genre: ", genre)
        download_list = get_genre_audio_info(genre, all_audio_info)
        count = 0
        for x in download_list:
            # print("Count before download: ", count)
            try:
                download_audio(x[0], genre, float(x[1]), float(x[2]), count=count)
                current_file = DIRECTORY_PATH + "\\" + genre + str(count) + ".wav"
                new_location = DATA_SET_PATH + "\\" + genre + "\\" + genre + str(count) + ".wav"
                shutil.move(current_file, new_location)
                count += 1
            except Exception:
                continue
            # current_file = DIRECTORY_PATH + "\\" + genre + str(count) + ".wav"
            # new_location = DATA_SET_PATH + "\\" + genre + "\\" + genre + str(count) + ".wav"
            # print("Current File: ", current_file)
            # print("Move to location: ", new_location)
            # shutil.move(current_file, new_location)
            # count += 1
            # print("Count after download: ", count)


    # count = 0
    # for id in hiphop_ids:
    #     for row in audio_info:
    #         if id in row[3]:
    #             download_audio(youtube_id=row[0], genre="hiphop", start=float(row[1]), end=float(row[2]), count=count)
    #             subprocess.call(["mv", os.getcwd + "\\" + ])
    #             count += 1
                
    
if __name__ == "__main__":
    main()