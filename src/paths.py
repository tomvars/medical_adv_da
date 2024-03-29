# Model Configuration
# replacing pretzel with bd0795ec38f7, change back accordingly
data_paths = {'dgx1': '/raid/tom/Training_ipmi_coronal',
              'tom-MS-7B09': '/data/tom/Training_ipmi_coronal',
              'dgx2-a.ai4vbh.local': '/raid/tomvars/Training_ipmi_coronal',
              'dgx1-1': '/raid/tomvars/Training_ipmi_coronal',
              'pretzel': '/raid/tom/Training_ipmi_coronal',
              'bd0795ec38f7': '/data2/tom/Training_ipmi_coronal'
              }

model_saving_paths = {'dgx1': '/raid/mauro/DART2019/',
                      'tom-MS-7B09': '/data/tom/MICCAI2020/',
                      'dgx2-a.ai4vbh.local': '/raid/tomvars/models/',
                      'dgx1-1': '/raid/tomvars/models',
                      'pretzel': '/raid/tom/Training_ipmi_coronal',
                      'bd0795ec38f7': '/data2/tom/DomainAdaptationJournal/models'
                      }

results_paths = {'dgx1': '/raid/mauro/DART2019/results',
                 'tom-MS-7B09': '/data/tom/MICCAI2020/results',
                 'dgx2-a.ai4vbh.local': '/raid/tomvars/results/',
                 'dgx1-1': '/raid/tomvars/results',
                 'pretzel': '/raid/tom/results',
                 'bd0795ec38f7': '/data2/tom/DomainAdaptationJournal/results'
                 }

inference_paths = {'dgx1': '/raid/mauro/DART2019/inference',
                 'tom-MS-7B09': '/data/tom/MICCAI2020/inference',
                 'dgx2-a.ai4vbh.local': '/raid/tomvars/inference/',
                 'dgx1-1': '/raid/tomvars/inference',
                 'pretzel': '/raid/tom/inference',
                 'bd0795ec38f7': '/data2/tom/DomainAdaptationJournal/inference'
                 }

tensorboard_paths = {'dgx1': '/raid/mauro/DART2019/tensorboard',
                 'tom-MS-7B09': '/data/tom/MICCAI2020/tensorboard',
                 'dgx2-a.ai4vbh.local': '/raid/tomvars/tensorboard/',
                 'dgx1-1': '/raid/tomvars/tensorboard',
                 'pretzel': '/raid/tom/tensorboard',
                 'bd0795ec38f7': '/data2/tom/DomainAdaptationJournal/tensorboard'
                 }

ms_path = {'dgx1':
               {
                'MS2008': '/raid/mauro/DATA/MS2008/skullstr/',
                'slovenia': '/raid/mauro/DATA/MSSelection/',
                'MSseg': '/raid/mauro/DATA/MSseg_2016/skullstr/',
                'clinic1': '/raid/mauro/DATA/MICCAI_2017/clinic1',
                'clinic2': '/raid/mauro/DATA/MICCAI_2017/clinic2',
                'clinic3': '/raid/mauro/DATA/MICCAI_2017/clinic3'},
           'tom-MS-7B09':
               {'test': '/data/tom/MICCAI2020/test/',
                'MS2008': '/data/tom/MICCAI2020/MS2008/skullstr/',
                'slovenia': '/data/tom/MICCAI2020/MSSelection/',
                'MSseg': '/data/tom/MICCAI2020/MSseg_2016/skullstr/',
                'clinic1': '/data/tom/MICCAI2020/MICCAI_2017_slices/clinic1',
                'clinic2': '/data/tom/MICCAI2020/MICCAI_2017_slices/clinic2',
                'clinic3': '/data/tom/MICCAI2020/MICCAI_2017_slices/clinic3'},
           'dgx2-a.ai4vbh.local':
               {'test': '/raid/tomvars/MICCAI2020/test/',
                'MS2008': '/raid/tomvars/MICCAI2020/MS2008/skullstr/',
                'slovenia': '/raid/tomvars/MICCAI2020/MSSelection/',
                'MSseg': '/raid/tomvars/MICCAI2020/MSseg_2016/skullstr/',
                'clinic1': '/raid/tomvars/MICCAI_2017_WMH_slices/clinic1',
                'clinic2': '/raid/tomvars/MICCAI_2017_WMH_slices/clinic2',
                'clinic3': '/raid/tomvars/MICCAI_2017_WMH_slices/clinic3',
                'clinic1_whole': '/raid/tomvars/MICCAI_2017/clinic1',
                'clinic2_whole': '/raid/tomvars/MICCAI_2017/clinic2',
                'clinic3_whole': '/raid/tomvars/MICCAI_2017/clinic3'
                },
           'dgx1-1':
               {'test': '/raid/tomvars/MICCAI2020/test/',
                'MS2008': '/raid/tomvars/MICCAI2020/MS2008/skullstr/',
                'slovenia': '/raid/tomvars/MICCAI2020/MSSelection/',
                'MSseg': '/raid/tomvars/MICCAI2020/MSseg_2016/skullstr/',
                'clinic1': '/raid/tomvars/MICCAI_2017_WMH_slices/clinic1',
                'clinic2': '/raid/tomvars/MICCAI_2017_WMH_slices/clinic2',
                'clinic3': '/raid/tomvars/MICCAI_2017_WMH_slices/clinic3',
                'clinic1_whole': '/raid/tomvars/MICCAI_2017/clinic1',
                'clinic2_whole': '/raid/tomvars/MICCAI_2017/clinic2',
                'clinic3_whole': '/raid/tomvars/MICCAI_2017/clinic3',
                'brats': '/raid/tomvars/Tumour/BRATS/slices',
                'brats_whole': '/raid/tomvars/Tumour/BRATS/whole_volume',
                'sheffield': '/raid/tomvars/Tumour/Sheffield/slices',
                'sheffield_whole': '/raid/tomvars/Tumour/Sheffield/whole',
                'sheffield_skullstr': '/raid/tomvars/Tumour/Sheffield/skullstr/slices',
                'sheffield_skullstr_whole': '/raid/tomvars/Tumour/Sheffield/skullstr/whole',
                'ms_combined': '/raid/tomvars/MS_combined/slices',
                'ms_combined_whole': '/raid/tomvars/MS_combined/whole',
                'isbi': '/raid/tomvars/ISBI2015/training/slices',
                'isbi_whole': '/raid/tomvars/ISBI2015/training/whole',
                },
           'pretzel': {
                'brats': '/raid/tom/BRATS/slices',
                'brats_whole': '/raid/tom/BRATS/whole_volume',
                'sheffield_skullstr': '/raid/tom/Sheffield/skullstr/slices',
                'sheffield_skullstr_whole': '/raid/tom/Sheffield/skullstr/whole',
                'ms_combined': '/raid/tom/MS_combined/slices',
                'ms_combined_whole': '/raid/tom/MS_combined/whole',
                'isbi': '/raid/tom/ISBI2015/training/slices',
                'isbi_whole': '/raid/tom/ISBI2015/training/whole',
                'isbi_test': '/raid/tom/ISBI2015/testing/slices',
                'isbi_test_whole': '/raid/tom/ISBI2015/testing/whole',
               'crossmoda_source': '/data2/tom/crossmoda/source_training/slices',
                'crossmoda_source_whole': '/data2/tom/crossmoda/source_training/whole',
                'crossmoda_target': '/data2/tom/crossmoda/target_training/slices',
                'crossmoda_target_whole': '/data2/tom/crossmoda/target_training/whole',
                },
           'bd0795ec38f7': {
                'brats': '/data2/tom/BRATS/slices',
                'brats_whole': '/data2/tom/BRATS/whole_volume',
                'sheffield_skullstr': '/data2/tom/Sheffield/skullstr/slices',
                'sheffield_skullstr_whole': '/data2/tom/Sheffield/skullstr/whole',
                'ms_combined': '/data2/tom/MS_combined/slices',
                'ms_combined_whole': '/data2/tom/MS_combined/whole',
                'isbi': '/data2/tom/ISBI2015/training/slices',
                'isbi_whole': '/data2/tom/ISBI2015/training/whole',
                'isbi_test': '/data2/tom/ISBI2015/testing/slices',
                'isbi_test_whole': '/data2/tom/ISBI2015/testing/whole',
                'crossmoda_source': '/data2/tom/crossmoda/source_training/slices',
                'crossmoda_source_whole': '/data2/tom/crossmoda/source_training/whole',
                'crossmoda_target': '/data2/tom/crossmoda/target_training/slices',
                'crossmoda_target_whole': '/data2/tom/crossmoda/target_training/whole',
                }
           }
DATASET_SPLITS = {'clinic1': 'dataset_split_clinic1.csv',
                  'clinic2': 'dataset_split_clinic2.csv',
                  'clinic3': 'dataset_split_clinic3.csv',
                  'isbi': 'dataset_split_isbi.csv',
                  'ms_combined': 'dataset_split_mscombined.csv',
                  'MS2008': 'dataset_split_MS2008.csv',
                  'MSSeg': 'dataset_split_MSSeg.csv',
                  'slovenia': 'dataset_split_slovenia.csv'}