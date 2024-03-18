'''
Evaluate the landmark results between the same scan but in different modalities (fat vs water). 

Author: Shuhao Lai <shuhaolai18@gmail.com>
'''

import re
import argparse

from collections import defaultdict


def parse_data_line(data):
    '''
    Extract x, y, z landmark predictions from a single localization prediction. 
    
    :param data: A single landmark prediction and other strings (for human readability)
    :return: Return the landmark x, y, z predictions as ints
    '''
    
    loc_pattern = 'Loc \((\d+), (\d+), (\d+)\)'
    match_obj = re.search(loc_pattern, data)
    x, y, z = match_obj.group(1), match_obj.group(2), match_obj.group(3)
    return int(x), int(y), int(z)


def parse_best_landmark_file(path):
    '''
    Extract the landmarks predictions from an entire file of predictions, which contains 
    predictions for water and fat scans. 
    
    :param path: Path to file of predictions. 
    :return: Prediction coordinates are stored in a data structure with patient ID as the 
        first index then scan type ('water' or 'fat') as the key. 
    '''
    
    with open(path) as file:
        patient_res = defaultdict(lambda: {})
        water_pattern = 'w_mri'
        fat_pattern = 'f_mri'
        id_pattern = 'md\d\d\d'
        for line in file:
            line = line.lower().strip()
            match_obj = re.search(id_pattern, line)
            patient_id = match_obj.group()
            patient_locs = []
            for i in range(9):
                loc = parse_data_line(next(file))
                patient_locs.append(loc)

            if re.search(water_pattern, line):
                patient_res[patient_id]['water'] = patient_locs
            elif re.search(fat_pattern, line):
                patient_res[patient_id]['fat'] = patient_locs
            else:
                raise ValueError('The file is not a water of fat MRI scan')
        return patient_res


def min_dist_best_landmark(landmark_path, output_path):
    '''
    Calculates the distance between the landmark predictions for a pair of water and fat MRI scans.
    A given pair of water and fat scans will be of the same patient. 
    
    :param landmark_path: Path containing the water and fat MRI landmarks
    :param output_path: Folder to save distance results. 
    '''
    
    def dist(x1, y1, z1, x2, y2, z2):
        return ((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2) ** 0.5

    patient_loc_map = parse_best_landmark_file(landmark_path)
    f = open(output_path, "w")
    for patient, locs_map in patient_loc_map.items():
        f.write(patient + '\n')
        if 'water' in locs_map and 'fat' in locs_map:
            for agent_i in range(9):
                x1, y1, z1 = locs_map['water'][agent_i]
                x2, y2, z2 = locs_map['fat'][agent_i]
                landmark_dist = dist(x1, y1, z1, x2, y2, z2)
                f.write(str(landmark_dist) + ' px\n')
        else:
            f.write('Patient does not have 2 sets of locations for each agent (SKIPPED)' + '\n')
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates the distance between the best landmarks for the water and fat MRI scans')
    parser.add_argument('-l', '--landmarks', required=True, help="Path containing the water and fat MRI landmarks")
    parser.add_argument('-o', '--output', required=True, help="Path of output file")
    args = parser.parse_args()

    min_dist_best_landmark(args.landmarks, args.output)