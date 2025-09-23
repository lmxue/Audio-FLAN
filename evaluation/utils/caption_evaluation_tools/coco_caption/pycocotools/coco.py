# =================================================================
# This code was pulled from https://github.com/tylin/coco-caption
# and refactored for Python 3.
# Image-specific names and comments were changed to be audio-specific.
# (not in the beginning comment)
# =================================================================

__author__ = 'tylin'
__version__ = '1.0.1'
# Interface for accessing the Microsoft COCO dataset.

# Microsoft COCO is a large image dataset designed for object detection,
# segmentation, and caption generation. pycocotools is a Python API that
# assists in loading, parsing and visualizing the annotations in COCO.
# Please visit http://mscoco.org/ for more information on COCO, including
# for the data, paper, and tutorials. The exact format of the annotations
# is also described on the COCO website. For example usage of the pycocotools
# please see pycocotools_demo.ipynb. In addition to this API, please download both
# the COCO images and annotations in order to run the demo.

# An alternative to using the API is to load the annotations directly
# into Python dictionary
# Using the API provides additional utility functions. Note that this API
# supports both *instance* and *caption* annotations. In the case of
# captions not all functions are defined (e.g. categories are undefined).

# The following API functions are defined:
#  COCO       - COCO api class that loads COCO annotation file and prepare data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  segToMask  - Convert polygon segmentation to binary mask.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load result file and create result api object.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
# Help on each functions can be accessed by: "help COCO>function".

# See also COCO>decodeMask,
# COCO>encodeMask, COCO>getAnnIds, COCO>getCatIds,
# COCO>getImgIds, COCO>loadAnns, COCO>loadCats,
# COCO>loadImgs, COCO>segToMask, COCO>showAnns

# Microsoft COCO Toolbox.      Version 1.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]

import json
import datetime
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
from skimage.draw import polygon
import copy

class COCO:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param audio_folder (str): location to the folder that hosts sounds.
        :return:
        """
        # load dataset
        self.dataset = {}
        self.anns = []
        self.audioToAnns = {}
        self.catToAudios = {}
        self.audios = []
        self.cats = []
        if not annotation_file == None:
            # print('loading annotations into memory...')
            time_t = datetime.datetime.utcnow()
            dataset = json.load(open(annotation_file, 'r'))
            # print(datetime.datetime.utcnow() - time_t)
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        # print('creating index...')
        audioToAnns = {ann['audio_id']: [] for ann in self.dataset['annotations']}
        anns =      {ann['id']:       [] for ann in self.dataset['annotations']}
        for ann in self.dataset['annotations']:
            audioToAnns[ann['audio_id']] += [ann]
            anns[ann['id']] = ann

        audios      = {aud['id']: {} for aud in self.dataset['audio samples']}
        for audio in self.dataset['audio samples']:
            audios[audio['id']] = audio

        cats = []
        catToAudios = []
        if self.dataset['type'] == 'instances':
            cats = {cat['id']: [] for cat in self.dataset['categories']}
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat
            catToAudios = {cat['id']: [] for cat in self.dataset['categories']}
            for ann in self.dataset['annotations']:
                catToAudios[ann['category_id']] += [ann['audio_id']]

        # print('index created!')

        # create class members
        self.anns = anns
        self.audioToAnns = audioToAnns
        self.catToAudios = catToAudios
        self.audios = audios
        self.cats = cats

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.datset['info'].items():
            print('%s: %s'%(key, value))

    def getAnnIds(self, audioIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param audioIds  (int array)     : get anns for given audios
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        audioIds = audioIds if type(audioIds) == list else [audioIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(audioIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(audioIds) == 0:
                anns = sum([self.audioToAnns[audioId] for audioId in audioIds if audioId in self.audioToAnns],[])
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if self.dataset['type'] == 'instances':
            if not iscrowd == None:
                ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
            else:
                ids = [ann['id'] for ann in anns]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if type(catNms) == list else [catNms]
        supNms = supNms if type(supNms) == list else [supNms]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name']          in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id']            in catIds]
        ids = [cat['id'] for cat in cats]
        return ids

    def getAudioIds(self, audioIds=[], catIds=[]):
        '''
        Get audio ids that satisfy given filter conditions.
        :param audioIds (int array) : get audios for given ids
        :param catIds (int array) : get audios with all given cats
        :return: ids (int array)  : integer array of audio ids
        '''
        audioIds = audioIds if type(audioIds) == list else [audioIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(audioIds) == len(catIds) == 0:
            ids = self.audios.keys()
        else:
            ids = set(audioIds)
            for catId in catIds:
                if len(ids) == 0:
                    ids = set(self.catToAudios[catId])
                else:
                    ids &= set(self.catToAudios[catId])
        return list(ids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if type(ids) == list:
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if type(ids) == list:
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadAudios(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying audio
        :return: audios (object array) : loaded audio objects
        """
        if type(ids) == list:
            return [self.audios[id] for id in ids]
        elif type(ids) == int:
            return [self.audios[ids]]

    def showAnns(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        if self.dataset['type'] == 'instances':
            ax = plt.gca()
            polygons = []
            color = []
            for ann in anns:
                c = np.random.random((1, 3)).tolist()[0]
                if type(ann['segmentation']) == list:
                    # polygon
                    for seg in ann['segmentation']:
                        poly = np.array(seg).reshape((len(seg)/2, 2))
                        polygons.append(Polygon(poly, True,alpha=0.4))
                        color.append(c)
                else:
                    # mask
                    mask = COCO.decodeMask(ann['segmentation'])
                    audio = np.ones( (mask.shape[0], mask.shape[1], 3) )
                    if ann['iscrowd'] == 1:
                        color_mask = np.array([2.0,166.0,101.0])/255
                    if ann['iscrowd'] == 0:
                        color_mask = np.random.random((1, 3)).tolist()[0]
                    for i in range(3):
                        audio[:,:,i] = color_mask[i]
                    ax.imshow(np.dstack( (audio, mask*0.5) ))
            p = PatchCollection(polygons, facecolors=color, edgecolors=(0,0,0,1), linewidths=3, alpha=0.4)
            ax.add_collection(p)
        if self.dataset['type'] == 'captions':
            for ann in anns:
                print(ann['caption'])

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCO()
        res.dataset['audio samples'] = [audio for audio in self.dataset['audio samples']]
        res.dataset['info'] = copy.deepcopy(self.dataset['info'])
        res.dataset['type'] = copy.deepcopy(self.dataset['type'])
        res.dataset['licenses'] = copy.deepcopy(self.dataset['licenses'])

        # print('Loading and preparing results...     ')
        time_t = datetime.datetime.utcnow()
        anns    = json.load(open(resFile))
        assert type(anns) == list, 'results in not an array of objects'
        annsAudioIds = [ann['audio_id'] for ann in anns]
        assert set(annsAudioIds) == (set(annsAudioIds) & set(self.getAudioIds())), \
               'Results do not correspond to current coco set'
        if 'caption' in anns[0]:
            audioIds = set([audio['id'] for audio in res.dataset['audio samples']]) & set([ann['audio_id'] for ann in anns])
            res.dataset['audio samples'] = [audio for audio in res.dataset['audio samples'] if audio['id'] in audioIds]
            for id, ann in enumerate(anns):
                ann['id'] = id
        elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
                ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2]*bb[3]
                ann['id'] = id
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                ann['area']=sum(ann['segmentation']['counts'][2:-1:2])
                ann['bbox'] = []
                ann['id'] = id
                ann['iscrowd'] = 0
        # print('DONE (t=%0.2fs)'%((datetime.datetime.utcnow() - time_t).total_seconds()))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res


    @staticmethod
    def decodeMask(R):
        """
        Decode binary mask M encoded via run-length encoding.
        :param   R (object RLE)    : run-length encoding of binary mask
        :return: M (bool 2D array) : decoded binary mask
        """
        N = len(R['counts'])
        M = np.zeros( (R['size'][0]*R['size'][1], ))
        n = 0
        val = 1
        for pos in range(N):
            val = not val
            for c in range(R['counts'][pos]):
                R['counts'][pos]
                M[n] = val
                n += 1
        return M.reshape((R['size']), order='F')

    @staticmethod
    def encodeMask(M):
        """
        Encode binary mask M using run-length encoding.
        :param   M (bool 2D array)  : binary mask to encode
        :return: R (object RLE)     : run-length encoding of binary mask
        """
        [h, w] = M.shape
        M = M.flatten(order='F')
        N = len(M)
        counts_list = []
        pos = 0
        # counts
        counts_list.append(1)
        diffs = np.logical_xor(M[0:N-1], M[1:N])
        for diff in diffs:
            if diff:
                pos +=1
                counts_list.append(1)
            else:
                counts_list[pos] += 1
        # if array starts from 1. start with 0 counts for 0
        if M[0] == 1:
            counts_list = [0] + counts_list
        return {'size':      [h, w],
               'counts':    counts_list ,
               }

    @staticmethod
    def segToMask( S, h, w ):
         """
         Convert polygon segmentation to binary mask.
         :param   S (float array)   : polygon segmentation mask
         :param   h (int)           : target mask height
         :param   w (int)           : target mask width
         :return: M (bool 2D array) : binary mask
         """
         M = np.zeros((h,w), dtype=np.bool)
         for s in S:
             N = len(s)
             rr, cc = polygon(np.array(s[1:N:2]), np.array(s[0:N:2])) # (y, x)
             M[rr, cc] = 1
         return M