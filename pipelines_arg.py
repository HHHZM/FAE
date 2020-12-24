
import os
import pickle
import pandas as pd
import csv
import numpy as np
from copy import deepcopy
import json
import argparse

from FAE.DataContainer.DataContainer import DataContainer
from FAE.FeatureAnalysis.IndexDict import Index2Dict
from FAE.FeatureAnalysis.Normalizer import NormalizerNone
from FAE.FeatureAnalysis.DimensionReduction import DimensionReductionByPCC
from FAE.Func.Metric import EstimatePrediction

from Utility.PathManager import MakeFolder
from Utility.Constants import *


class PipelinesManager(object):
    def __init__(self, balancer=None, normalizer_list=[NormalizerNone],
                 dimension_reduction_list=[DimensionReductionByPCC()], feature_selector_list=[],
                 feature_selector_num_list=[], classifier_list=[],
                 cross_validation=None, is_hyper_parameter=False, logger=None):
        self.balance = balancer
        self.normalizer_list = normalizer_list
        self.dimension_reduction_list = dimension_reduction_list
        self.feature_selector_list = feature_selector_list
        self.feature_selector_num_list = feature_selector_num_list
        self.classifier_list = classifier_list
        self.cv = cross_validation
        self.is_hyper_parameter = is_hyper_parameter
        self.__logger = logger
        self.version = VERSION

        self.total_metric = {TRAIN: pd.DataFrame(columns=HEADER),
                             BALANCE_TRAIN: pd.DataFrame(columns=HEADER),
                             TEST: pd.DataFrame(columns=HEADER),
                             CV_TRAIN: pd.DataFrame(columns=HEADER),
                             CV_VAL: pd.DataFrame(columns=HEADER)}

        self.GenerateAucDict()

    def SaveAll(self, store_folder):
        self.SaveAucDict(store_folder)
        self.SavePipelineInfo(store_folder)

    def SavePipelineInfo(self, store_folder):
        with open(os.path.join(store_folder, 'pipeline_info.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow([VERSION_NAME, self.version])
            writer.writerow([CROSS_VALIDATION, self.cv.GetName()])
            writer.writerow([BALANCE, self.balance.GetName()])
            writer.writerow([NORMALIER] + [one.GetName() for one in self.normalizer_list])
            writer.writerow([DIMENSION_REDUCTION] + [one.GetName() for one in self.dimension_reduction_list])
            writer.writerow([FEATURE_SELECTOR] + [one.GetName() for one in self.feature_selector_list])
            writer.writerow([FEATURE_NUMBER] + self.feature_selector_num_list)
            writer.writerow([CLASSIFIER] + [one.GetName() for one in self.classifier_list])

    def SaveAucDict(self, store_folder):
        with open(os.path.join(store_folder, 'auc_metric.pkl'), 'wb') as file:
            pickle.dump(self.__auc_dict, file, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(store_folder, 'auc_std_metric.pkl'), 'wb') as file:
            pickle.dump(self.__auc_std_dict, file, pickle.HIGHEST_PROTOCOL)

    def LoadAll(self, store_folder):
        return self.LoadAucDict(store_folder) and self.LoadPipelineInfo(store_folder)



    def GetRealFeatureNum(self, store_folder):
        files = os.listdir(store_folder)
        for file in files:
            if file.find('_features.csv') != -1:
                feature_file = os.path.join(store_folder, file)
                pdf = pd.read_csv(feature_file)
                return len(pdf.columns) - 1
        return 0


    def LoadPipelineInfo(self, store_folder):
        index_2_dict = Index2Dict()
        info_path = os.path.join(store_folder, 'pipeline_info.csv')
        if not os.path.exists(info_path):
            return False

        with open(info_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == VERSION_NAME:
                    self.version = row[1]
                    if self.version not in ACCEPT_VERSION:
                        return False
                elif row[0] == CROSS_VALIDATION:
                    self.cv = index_2_dict.GetInstantByIndex(row[1])
                elif row[0] == BALANCE:
                    self.balance = index_2_dict.GetInstantByIndex(row[1])
                elif row[0] == NORMALIER:
                    self.normalizer_list = [index_2_dict.GetInstantByIndex(index) for index in row[1:]]
                elif row[0] == DIMENSION_REDUCTION:
                    self.dimension_reduction_list = [index_2_dict.GetInstantByIndex(index) for index in row[1:]]
                elif row[0] == FEATURE_SELECTOR:
                    self.feature_selector_list = [index_2_dict.GetInstantByIndex(index) for index in row[1:]]
                elif row[0] == FEATURE_NUMBER:
                    feature_number = self.GetRealFeatureNum(store_folder)
                    number = len(row) - 1 if len(row) - 1 > feature_number else feature_number
                    self.feature_selector_num_list = row[1: number]
                elif row[0] == CLASSIFIER:
                    self.classifier_list = [index_2_dict.GetInstantByIndex(index) for index in row[1:]]
                else:
                    print('Unknown name: {}'.format(row[0]))
                    raise KeyError
        return True

    def LoadAucDict(self, store_folder):
        auc_path = os.path.join(store_folder, 'auc_metric.pkl')
        std_path = os.path.join(store_folder, 'auc_std_metric.pkl')
        if not(os.path.exists(auc_path) and os.path.exists(std_path)):
            return False

        with open(auc_path, 'rb') as file:
            self.__auc_dict = pickle.load(file)
        with open(std_path, 'rb') as file:
            self.__auc_std_dict = pickle.load(file)

        return True

    def SaveOneResult(self, pred, label, key_name, case_name, matric_indexs, model_name,
                      store_root='', model_folder=''):
        assert(len(matric_indexs) == 5)
        norm_index, dr_index, fs_index, fn_index, cls_index = matric_indexs

        info = pd.DataFrame({'Pred': pred, 'Label': label}, index=case_name)
        metric = EstimatePrediction(pred, label, key_name)

        self.__auc_dict[key_name][norm_index, dr_index, fs_index, fn_index, cls_index] = \
            metric['{}_{}'.format(key_name, AUC)]
        self.__auc_std_dict[key_name][norm_index, dr_index, fs_index, fn_index, cls_index] = \
            metric['{}_{}'.format(key_name, AUC_STD)]

        if store_root:
            info.to_csv(os.path.join(model_folder, '{}_prediction.csv'.format(key_name)))
            self._AddOneMetric(metric, os.path.join(model_folder, 'metrics.csv'))
            self._MergeOneMetric(metric, key_name, model_name)

    def _AddOneMetric(self, info, store_path):
        if not os.path.exists(store_path):
            df = pd.DataFrame(info, index=['Value']).T
            df.to_csv(store_path)
        else:
            df = pd.read_csv(store_path, index_col=0)
            new_df = pd.DataFrame(info, index=['Value']).T
            df = pd.concat((df, new_df), sort=True, axis=0)
            df.to_csv(store_path)

    def _MergeOneMetric(self, metric, key, model_name):
        save_info = [metric['{}_{}'.format(key, index)] for index in HEADER]
        self.total_metric[key].loc[model_name] = save_info

    def _AddOneCvPrediction(self, store_path, prediction):
        if not os.path.exists(store_path):
            prediction.to_csv(store_path)
        else:
            temp = pd.read_csv(store_path, index_col=0)
            temp = pd.concat((temp, prediction), axis=0)
            temp.to_csv(store_path)

    def GenerateAucDict(self):
        self.total_num = len(self.normalizer_list) * \
                         len(self.dimension_reduction_list) * \
                         len(self.feature_selector_list) * \
                         len(self.classifier_list) * \
                         len(self.feature_selector_num_list)

        try:
            matrix = np.zeros((len(self.normalizer_list), len(self.dimension_reduction_list),
                               len(self.feature_selector_list), len(self.feature_selector_num_list),
                               len(self.classifier_list)))
        except:
            matrix = np.zeros(())

        self.__auc_dict = {CV_TRAIN: deepcopy(matrix), CV_VAL: deepcopy(matrix), TEST: deepcopy(matrix),
                           TRAIN: deepcopy(matrix), BALANCE_TRAIN: deepcopy(matrix)}
        self.__auc_std_dict = deepcopy(self.__auc_dict)

    def GetAuc(self):
        return self.__auc_dict

    def GetAucStd(self):
        return self.__auc_std_dict

    def GetStoreName(self, normalizer_name='', dimension_rediction_name='', feature_selector_name='',
                     feature_number='', classifer_name=''):
        case_name = '{}_{}_{}_{}_{}'.format(
            normalizer_name, dimension_rediction_name, feature_selector_name, feature_number, classifer_name
        )
        return case_name

    def SplitFolder(self, pipeline_name, store_root):
        normalizer, dr, fs, fn, cls = pipeline_name.split('_')

        normalizer_folder = os.path.join(store_root, normalizer)
        dr_folder = os.path.join(normalizer_folder, dr)
        fs_folder = os.path.join(dr_folder, '{}_{}'.format(fs, fn))
        cls_folder = os.path.join(fs_folder, cls)

        assert(os.path.isdir(store_root) and os.path.isdir(normalizer_folder) and os.path.isdir(dr_folder) and
               os.path.isdir(fs_folder) and os.path.isdir(cls_folder))
        return normalizer_folder, dr_folder, fs_folder, cls_folder


    def RunWithCV(self, train_container, store_folder=''):
        for group, containers in enumerate(self.cv.Generate(train_container)):
            cv_train_container, cv_val_container = containers

            balance_cv_train_container = self.balance.Run(cv_train_container)
            num = 0
            for norm_index, normalizer in enumerate(self.normalizer_list):
                norm_store_folder = MakeFolder(store_folder, normalizer.GetName())
                norm_cv_train_container = normalizer.Run(balance_cv_train_container)
                norm_cv_val_container = normalizer.Transform(cv_val_container)

                for dr_index, dr in enumerate(self.dimension_reduction_list):
                    dr_store_folder = MakeFolder(norm_store_folder, dr.GetName())
                    if dr:
                        dr_cv_train_container = dr.Run(norm_cv_train_container)
                        dr_cv_val_container = dr.Transform(norm_cv_val_container)
                    else:
                        dr_cv_train_container = norm_cv_train_container
                        dr_cv_val_container = norm_cv_val_container

                    for fs_index, fs in enumerate(self.feature_selector_list):

                        # 由于效率问题，BC单独拿出来算
                        if fs.GetName() == 'BC':
                            fs.ClearFoldResult()
                            fs.SetSelectedFeatureNumber(max(self.feature_selector_num_list))
                            fs.PreRun(dr_cv_train_container)

                        for fn_index, fn in enumerate(self.feature_selector_num_list):
                            if fs:
                                fs_store_folder = MakeFolder(dr_store_folder, '{}_{}'.format(fs.GetName(), fn))
                                fs.SetSelectedFeatureNumber(fn)
                                fs_cv_train_container = fs.Run(dr_cv_train_container)
                                fs_cv_val_container = fs.Transform(dr_cv_val_container)
                            else:
                                fs_store_folder = dr_store_folder
                                fs_cv_train_container = dr_cv_train_container
                                fs_cv_val_container = dr_cv_val_container

                            for cls_index, cls in enumerate(self.classifier_list):
                                cls_store_folder = MakeFolder(fs_store_folder, cls.GetName())
                                model_name = self.GetStoreName(normalizer.GetName(),
                                                               dr.GetName(),
                                                               fs.GetName(),
                                                               str(fn),
                                                               cls.GetName())
                                num += 1
                                yield self.total_num, num, group

                                cls.SetDataContainer(fs_cv_train_container)
                                cls.Fit()

                                cv_train_pred = cls.Predict(fs_cv_train_container.GetArray())
                                cv_train_label = fs_cv_train_container.GetLabel()
                                cv_train_info = pd.DataFrame({'Pred': cv_train_pred, 'Label': cv_train_label,
                                                              'Group': [group for temp in cv_train_label]},
                                                             index=fs_cv_train_container.GetCaseName())

                                cv_val_pred = cls.Predict(fs_cv_val_container.GetArray())
                                cv_val_label = fs_cv_val_container.GetLabel()
                                cv_val_info = pd.DataFrame({'Pred': cv_val_pred, 'Label': cv_val_label,
                                                            'Group': [group for temp in cv_val_label]},
                                                           index=fs_cv_val_container.GetCaseName())

                                if store_folder:
                                    self._AddOneCvPrediction(os.path.join(cls_store_folder,
                                                                         '{}_prediction.csv'.format(CV_TRAIN)),
                                                             cv_train_info)
                                    self._AddOneCvPrediction(os.path.join(cls_store_folder,
                                                                         '{}_prediction.csv'.format(CV_VAL)),
                                                             cv_val_info)


    def RunWithCV_fold(self, train_container, fold, store_folder=''):
        for group, containers in enumerate(self.cv.Generate(train_container)):
            
            if group != fold:
                continue

            cv_train_container, cv_val_container = containers

            balance_cv_train_container = self.balance.Run(cv_train_container)
            num = 0
            for norm_index, normalizer in enumerate(self.normalizer_list):
                norm_store_folder = MakeFolder(store_folder, normalizer.GetName())

                # 在未执行类别平衡的数据集进行normalizetion
                norm_cv_train_container_nonebalance = normalizer.Run(cv_train_container)

                norm_cv_train_container = normalizer.Run(balance_cv_train_container)
                norm_cv_val_container = normalizer.Transform(cv_val_container)

                for dr_index, dr in enumerate(self.dimension_reduction_list):
                    dr_store_folder = MakeFolder(norm_store_folder, dr.GetName())
                    if dr:
                        dr_cv_train_container = dr.Run(norm_cv_train_container)
                        dr_cv_val_container = dr.Transform(norm_cv_val_container)
                        # 未平衡
                        dr_cv_val_container_nonebalance = dr.Transform(norm_cv_train_container_nonebalance)
                    else:
                        dr_cv_train_container = norm_cv_train_container
                        dr_cv_val_container = norm_cv_val_container
                        # 未平衡
                        dr_cv_val_container_nonebalance = norm_cv_train_container_nonebalance

                    for fs_index, fs in enumerate(self.feature_selector_list):

                        # 由于效率问题，BC单独拿出来算
                        if fs.GetName() == 'BC':
                            fs.ClearFoldResult()
                            fs.SetSelectedFeatureNumber(max(self.feature_selector_num_list))
                            if fs.target == 'balance':
                                fs.PreRun(dr_cv_train_container)
                            # 设置为在未进行SMOTE数据平衡的情况下进行BC特征选择
                            elif fs.target == 'nonebalance':
                                fs.PreRun(dr_cv_val_container_nonebalance)

                        for fn_index, fn in enumerate(self.feature_selector_num_list):
                            if fs:
                                fs_store_folder = MakeFolder(dr_store_folder, '{}_{}'.format(fs.GetName(), fn))
                                fs.SetSelectedFeatureNumber(fn)
                                fs_cv_train_container = fs.Run(dr_cv_train_container)
                                fs_cv_val_container = fs.Transform(dr_cv_val_container)
                            else:
                                fs_store_folder = dr_store_folder
                                fs_cv_train_container = dr_cv_train_container
                                fs_cv_val_container = dr_cv_val_container

                            for cls_index, cls in enumerate(self.classifier_list):
                                cls_store_folder = MakeFolder(fs_store_folder, cls.GetName())
                                model_name = self.GetStoreName(normalizer.GetName(),
                                                               dr.GetName(),
                                                               fs.GetName(),
                                                               str(fn),
                                                               cls.GetName())
                                num += 1
                                print(self.total_num, num, group)

                                cls.SetDataContainer(fs_cv_train_container)
                                cls.Fit()

                                cv_train_pred = cls.Predict(fs_cv_train_container.GetArray())
                                cv_train_label = fs_cv_train_container.GetLabel()
                                cv_train_info = pd.DataFrame({'Pred': cv_train_pred, 'Label': cv_train_label,
                                                              'Group': [group for temp in cv_train_label]},
                                                             index=fs_cv_train_container.GetCaseName())

                                cv_val_pred = cls.Predict(fs_cv_val_container.GetArray())
                                cv_val_label = fs_cv_val_container.GetLabel()
                                cv_val_info = pd.DataFrame({'Pred': cv_val_pred, 'Label': cv_val_label,
                                                            'Group': [group for temp in cv_val_label]},
                                                           index=fs_cv_val_container.GetCaseName())

                                if store_folder:
                                    self._AddOneCvPrediction(os.path.join(cls_store_folder,
                                                                         '{}_prediction.csv'.format(CV_TRAIN)),
                                                             cv_train_info)
                                    self._AddOneCvPrediction(os.path.join(cls_store_folder,
                                                                         '{}_prediction.csv'.format(CV_VAL)),
                                                             cv_val_info)


    def MergeCvResult(self, store_folder):
        num = 0
        for norm_index, normalizer in enumerate(self.normalizer_list):
            norm_store_folder = MakeFolder(store_folder, normalizer.GetName())
            for dr_index, dr in enumerate(self.dimension_reduction_list):
                dr_store_folder = MakeFolder(norm_store_folder, dr.GetName())
                for fs_index, fs in enumerate(self.feature_selector_list):
                    for fn_index, fn in enumerate(self.feature_selector_num_list):
                        fs_store_folder = MakeFolder(dr_store_folder, '{}_{}'.format(fs.GetName(), fn))
                        for cls_index, cls in enumerate(self.classifier_list):
                            cls_store_folder = MakeFolder(fs_store_folder, cls.GetName())
                            model_name = self.GetStoreName(normalizer.GetName(),
                                                           dr.GetName(),
                                                           fs.GetName(),
                                                           str(fn),
                                                           cls.GetName())
                            num += 1
                            yield self.total_num, num

                            # ADD CV Train
                            cv_train_info = pd.read_csv(os.path.join(cls_store_folder,
                                                                     '{}_prediction.csv'.format(CV_TRAIN)),
                                                        index_col=0)
                            cv_train_metric = EstimatePrediction(cv_train_info['Pred'], cv_train_info['Label'],
                                                                 key_word=CV_TRAIN)
                            self.__auc_dict[CV_TRAIN][norm_index, dr_index, fs_index, fn_index, cls_index] = \
                                cv_train_metric['{}_{}'.format(CV_TRAIN, AUC)]
                            self.__auc_std_dict[CV_TRAIN][norm_index, dr_index, fs_index, fn_index, cls_index] = \
                                cv_train_metric['{}_{}'.format(CV_TRAIN, AUC_STD)]
                            self._AddOneMetric(cv_train_metric, os.path.join(cls_store_folder, 'metrics.csv'))
                            self._MergeOneMetric(cv_train_metric, CV_TRAIN, model_name)

                            # ADD CV Validation
                            cv_val_info = pd.read_csv(os.path.join(cls_store_folder,
                                                                   '{}_prediction.csv'.format(CV_VAL)),
                                                      index_col=0)
                            cv_val_metric = EstimatePrediction(cv_val_info['Pred'], cv_val_info['Label'],
                                                               key_word=CV_VAL)
                            self.__auc_dict[CV_VAL][norm_index, dr_index, fs_index, fn_index, cls_index] = \
                                cv_val_metric['{}_{}'.format(CV_VAL, AUC)]
                            self.__auc_std_dict[CV_VAL][norm_index, dr_index, fs_index, fn_index, cls_index] = \
                                cv_val_metric['{}_{}'.format(CV_VAL, AUC_STD)]
                            self._AddOneMetric(cv_val_metric, os.path.join(cls_store_folder, 'metrics.csv'))
                            self._MergeOneMetric(cv_val_metric, CV_VAL, model_name)

        self.total_metric[CV_TRAIN].to_csv(os.path.join(store_folder, '{}_results.csv'.format(CV_TRAIN)))
        self.total_metric[CV_VAL].to_csv(os.path.join(store_folder, '{}_results.csv'.format(CV_VAL)))


    def RunWithCV_fold_merge(self, train_container, store_folder=''):
        store_folder_target = MakeFolder(store_folder, 'cross_validation')

        for group, containers in enumerate(self.cv.Generate(train_container)):
            # 当前fold的路径
            store_folder_fold = MakeFolder(store_folder, 'fold_' + str(group))

            for norm_index, normalizer in enumerate(self.normalizer_list):
                norm_store_folder = MakeFolder(store_folder_fold, normalizer.GetName())
                norm_store_folder_t = MakeFolder(store_folder_target, normalizer.GetName())
                
                for dr_index, dr in enumerate(self.dimension_reduction_list):
                    dr_store_folder = MakeFolder(norm_store_folder, dr.GetName())
                    dr_store_folder_t = MakeFolder(norm_store_folder_t, dr.GetName())

                    for fs_index, fs in enumerate(self.feature_selector_list):

                        for fn_index, fn in enumerate(self.feature_selector_num_list):
                            if fs:
                                fs_store_folder = MakeFolder(dr_store_folder, '{}_{}'.format(fs.GetName(), fn))
                                fs_store_folder_t = MakeFolder(dr_store_folder_t, '{}_{}'.format(fs.GetName(), fn))
                            else:
                                fs_store_folder = dr_store_folder
                                fs_store_folder_t = dr_store_folder_t

                            for cls_index, cls in enumerate(self.classifier_list):
                                cls_store_folder = MakeFolder(fs_store_folder, cls.GetName())
                                cls_store_folder_t = MakeFolder(fs_store_folder_t, cls.GetName())

                                # model_name = self.GetStoreName(normalizer.GetName(),
                                #                                dr.GetName(),
                                #                                fs.GetName(),
                                #                                str(fn),
                                #                                cls.GetName())

                                cv_train_info = pd.read_csv(os.path.join(cls_store_folder, '{}_prediction.csv'.format(CV_TRAIN)), index_col=0)
                                cv_val_info = pd.read_csv(os.path.join(cls_store_folder, '{}_prediction.csv'.format(CV_VAL)), index_col=0)
                                
                                self._AddOneCvPrediction(os.path.join(cls_store_folder_t,
                                                                        '{}_prediction.csv'.format(CV_TRAIN)),
                                                            cv_train_info)
                                self._AddOneCvPrediction(os.path.join(cls_store_folder_t,
                                                                        '{}_prediction.csv'.format(CV_VAL)),
                                                            cv_val_info)
        
        return store_folder_target


def main(args):

    balancer = args['balancer']
    normalizer_list = args['normalizer_list']
    dimension_reduction_list = args['dimension_reduction_list']
    feature_selector_list = args['feature_selector_list']
    feature_selector_args_dict = args['feature_selector_args_dict']
    feature_selector_num_list = args['feature_selector_num_list']
    classifier_list = args['classifier_list']
    cross_validation = args['cross_validation']
    fold_list = args["fold_list"]

    train_csv = args['train_csv']
    # test_csv = args['test_csv']
    store_folder_path = args['store_folder_path']

    train = DataContainer()
    train.Load(train_csv)

    if os.path.exists(store_folder_path):
        os.system('rm -r ' + store_folder_path)
    os.system('mkdir ' + store_folder_path)

    manager = PipelinesManager()
    index_dict = Index2Dict()

    faps = PipelinesManager(balancer=index_dict.GetInstantByIndex(balancer),
                            normalizer_list=[index_dict.GetInstantByIndex(i) for i in normalizer_list],
                            dimension_reduction_list=[index_dict.GetInstantByIndex(i) for i in dimension_reduction_list],
                            feature_selector_list=[index_dict.GetInstantByIndex(i) for i in feature_selector_list],
                            feature_selector_num_list=[i for i in feature_selector_num_list],
                            classifier_list=[index_dict.GetInstantByIndex(i) for i in classifier_list],
                            cross_validation=index_dict.GetInstantByIndex(cross_validation))

    # 方便设置BC的参数
    for feature_selector in faps.feature_selector_list:
        if feature_selector.GetName() in feature_selector_args_dict.keys():
            for k, v in feature_selector_args_dict[feature_selector.GetName()].items():
                exec('feature_selector.' + k + ' = v')

    # for total, num, group in faps.RunWithCV(train, store_folder=store_folder_path):
    #     print(total, num, group)

    for fold in fold_list:
        store_folder_path_fold = os.path.join(store_folder_path, 'fold_' + str(fold))
        os.system('mkdir ' + store_folder_path_fold)
        faps.RunWithCV_fold(train, fold=fold, store_folder=store_folder_path_fold)

    cv_merge_folder = faps.RunWithCV_fold_merge(train, store_folder=store_folder_path)

    for total, num in faps.MergeCvResult(store_folder=cv_merge_folder):
        print(total, num)

    return 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='args.json')
    args_config = parser.parse_args()

    with open(args_config.config, 'r') as f:
        args = json.load(f)

    main(args)

    
