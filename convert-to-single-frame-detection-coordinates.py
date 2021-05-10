import os
from absl import app, flags, logging
from absl.flags import FLAGS
FLAGS = flags.FLAGS
flags.DEFINE_string('input_file_txt','','Input text file')
flags.DEFINE_string('output_folder','','Input text file')



def main(_argv):
    with open(FLAGS.input_file_txt, 'r') as f:
        lines = f.readlines()

    output_folder_path = FLAGS.output_folder
    current_number_on_start_of_line = None
    print(len(lines))
    for line in lines:
        # get first char from line
        firstCharacter = line[:1]
        if current_number_on_start_of_line != firstCharacter:
            current_number_on_start_of_line = firstCharacter
            file_output = open(output_folder_path + f'{int(firstCharacter):06}','w')
            file_output.write(line)
        else:
            file_output = open(output_folder_path + f'{int(firstCharacter):06}','w')
            file_output.write(line)


        