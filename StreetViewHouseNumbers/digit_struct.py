import h5py
import numpy as np
import data_loader
import os

class DigitStruct:

    directory = ""
    labels_path = ""

    def __init__(self, file):
        self.file = h5py.File(file, 'r')
        self.digit_struct_name = self.file['digitStruct']['name']
        self.digit_struct_bbox = self.file['digitStruct']['bbox']
        self.directory = os.path.dirname(file)

    def load_labels_and_paths(self):
        file_path = os.path.join(self.directory, "labels_and_paths.pickle")

        result = data_loader.openPickle(file_path)
        if result != None:
            return result

        print("We need to extract the labels from the structure file. This will take a few minutes...")
        num_labels = 11
        max_digits = 5
        all_structs = self.get_all_imgs_and_digit_structure()

        labels = []
        paths = []
        for i in range(0, len(all_structs)):
            struct = all_structs[i]
            label_numbers = struct["label"]

            if(len(label_numbers) > max_digits):
                print("Ignoring element with more than five digits", i)
                print(struct)
                continue
            new_labels = []
            #Add length as first item
            one_hot_length = (np.arange(max_digits) == len(label_numbers)).astype(np.float32)
            new_labels.append(one_hot_length)

            for j in range(0, len(label_numbers)):
                label = int(label_numbers[j])
                newLabel = (np.arange(num_labels) == label).astype(np.float32)
                new_labels.append(newLabel)
            for j in range(len(label_numbers), max_digits):
                newLabel = (np.zeros(num_labels))
                newLabel[-1] = 1.0
                new_labels.append(newLabel)

            labels.append(new_labels)
            paths.append(struct["name"])

        labels = np.array(labels)
        paths = np.array(paths)
        data_loader.savePickle((labels,paths), file_path)
        return (labels, paths)

    def get_img_name(self, n):
        '''
            accepts: index for digit structure
            returns: the 'name' string for for the index in digitStruct.
            ie: "2.png"
        '''
        name = ''.join([chr(c[0]) for c in self.file[self.digit_struct_name[n][0]].value])
        return name

    def bbox_helper(self, attr):
        '''bbox_helper abstracts the bbox or an array of bbox.
           used internally with get_bbox
        '''
        if (len(attr) > 1):
            attr = [self.file[attr.value[j].item()].value[0][0] for j in range(len(attr))]
        else:
            attr = [attr.value[0][0]]
        return attr

    def get_bbox(self, n):
        '''getBbox returns a dict of data for the n(th) bbox. 
          accepts: index for digit structure
          returns: a hash with the coordiantes for the bbox
          ie {'width': [23.0, 26.0], 'top': [29.0, 25.0], 'label': [2.0, 3.0], 'left': [77.0, 98.0], 'height': [32.0, 32.0]}
        '''
        bbox = {}
        bb = self.digit_struct_bbox[n].item()

        bbox['label'] = self.bbox_helper(self.file[bb]["label"])
        bbox['top'] = self.bbox_helper(self.file[bb]["top"])
        bbox['left'] = self.bbox_helper(self.file[bb]["left"])
        bbox['height'] = self.bbox_helper(self.file[bb]["height"])
        bbox['width'] = self.bbox_helper(self.file[bb]["width"])

        return bbox

    def get_digit_structure(self, n):
        structure = self.get_bbox(n)
        structure['name'] = self.get_img_name(n)
        return structure

    def get_all_imgs_and_digit_structure(self):
        structs = []
        for i in range(len(self.digit_struct_name)):
            structs.append(self.get_digit_structure(i))
        print("Done extract")
        return structs
