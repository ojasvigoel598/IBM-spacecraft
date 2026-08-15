# MissionMind - Full ML Model Metrics Report

_Generated from `ml/compare.py` (no-leakage protocol: supervised trained on t<2500, evaluated on full files + t>=2500 holdouts). Classification metrics from `ml/metrics.py`. MSE/RMSE are not computed here because the task is anomaly detection (binary classification); the autoencoder-family models (MLP Autoencoder, Hybrid DIF, PINN) natively use **reconstruction error** (MSE per sample) as their anomaly score._

## IsolationForest (Baseline Unsupervised)

| metric | solar_failure | radiator_failure | normal | solar_holdout | radiator_holdout |
|---|---|---|---|---|---|
| accuracy | 0.8742 | 0.1156 | 0.9489 | 1.0000 | 0.0000 |
| precision | 0.9369 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| recall | 0.9103 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| f1 | 0.9234 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| balanced_accuracy | 0.8018 | 0.3467 | 0.9489 | 1.0000 | 0.0000 |
| mcc | 0.5732 | -0.5190 | 0.0000 | 0.0000 | 0.0000 |
| tn | 416 | 416 | 3416 | 0 | 0 |
| fp | 184 | 184 | 184 | 0 | 0 |
| fn | 269 | 3000 | 0 | 0 | 1100 |
| tp | 2731 | 0 | 0 | 1100 | 0 |
| fpr | 0.3067 | 0.3067 | 0.0511 | 0.0000 | 0.0000 |
| fnr | 0.0897 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| roc_auc | 0.8636 | 0.3919 | nan | nan | nan |
| pr_auc | 0.9253 | 0.8320 | nan | nan | nan |
| fpr_before_600 | 0.3067 | 0.3067 | 0.3067 | 0.0000 | 0.0000 |
| tpr_after_900 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| detection_delay_s | 269.0000 | 3600.0000 | 3600.0000 | 1900.0000 | 3600.0000 |
| early_detection_rate_600_900 | 0.1063 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| early_detected | True | False | False | False | False |
| first_detection_time | 869.0000 | � | � | 2500.0000 | � |
| mtd_after_end_s | 0.0000 | 3600.0000 | 3600.0000 | 1600.0000 | 3600.0000 |

## LOF (Unsupervised)

| metric | solar_failure | radiator_failure | normal | solar_holdout | radiator_holdout |
|---|---|---|---|---|---|
| accuracy | 0.9947 | 0.9925 | 0.9956 | 1.0000 | 1.0000 |
| precision | 0.9947 | 0.9947 | 0.0000 | 1.0000 | 1.0000 |
| recall | 0.9990 | 0.9963 | 0.0000 | 1.0000 | 1.0000 |
| f1 | 0.9968 | 0.9955 | 0.0000 | 1.0000 | 1.0000 |
| balanced_accuracy | 0.9862 | 0.9848 | 0.9956 | 1.0000 | 1.0000 |
| mcc | 0.9809 | 0.9729 | 0.0000 | 0.0000 | 0.0000 |
| tn | 584 | 584 | 3584 | 0 | 0 |
| fp | 16 | 16 | 16 | 0 | 0 |
| fn | 3 | 11 | 0 | 0 | 0 |
| tp | 2997 | 2989 | 0 | 1100 | 1100 |
| fpr | 0.0267 | 0.0267 | 0.0044 | 0.0000 | 0.0000 |
| fnr | 0.0010 | 0.0037 | 0.0000 | 0.0000 | 0.0000 |
| roc_auc | 0.9996 | 0.9997 | nan | nan | nan |
| pr_auc | 0.9999 | 0.9999 | nan | nan | nan |
| fpr_before_600 | 0.0267 | 0.0267 | 0.0267 | 0.0000 | 0.0000 |
| tpr_after_900 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| detection_delay_s | 3.0000 | 11.0000 | 3600.0000 | 1900.0000 | 1900.0000 |
| early_detection_rate_600_900 | 0.9900 | 0.9635 | 0.0000 | 0.0000 | 0.0000 |
| early_detected | True | True | False | False | False |
| first_detection_time | 603.0000 | 611.0000 | � | 2500.0000 | 2500.0000 |
| mtd_after_end_s | 0.0000 | 0.0000 | 3600.0000 | 1600.0000 | 1600.0000 |

## OneClassSVM (Unsupervised)

| metric | solar_failure | radiator_failure | normal | solar_holdout | radiator_holdout |
|---|---|---|---|---|---|
| accuracy | 0.9850 | 0.9764 | 0.9628 | 1.0000 | 1.0000 |
| precision | 0.9833 | 0.9831 | 0.0000 | 1.0000 | 1.0000 |
| recall | 0.9990 | 0.9887 | 0.0000 | 1.0000 | 1.0000 |
| f1 | 0.9911 | 0.9859 | 0.0000 | 1.0000 | 1.0000 |
| balanced_accuracy | 0.9570 | 0.9518 | 0.9628 | 1.0000 | 1.0000 |
| mcc | 0.9454 | 0.9142 | 0.0000 | 0.0000 | 0.0000 |
| tn | 549 | 549 | 3466 | 0 | 0 |
| fp | 51 | 51 | 134 | 0 | 0 |
| fn | 3 | 34 | 0 | 0 | 0 |
| tp | 2997 | 2966 | 0 | 1100 | 1100 |
| fpr | 0.0850 | 0.0850 | 0.0372 | 0.0000 | 0.0000 |
| fnr | 0.0010 | 0.0113 | 0.0000 | 0.0000 | 0.0000 |
| roc_auc | 0.9996 | 0.9961 | nan | nan | nan |
| pr_auc | 0.9999 | 0.9990 | nan | nan | nan |
| fpr_before_600 | 0.0850 | 0.0850 | 0.0850 | 0.0000 | 0.0000 |
| tpr_after_900 | 1.0000 | 1.0000 | 0.0308 | 1.0000 | 1.0000 |
| detection_delay_s | 3.0000 | 34.0000 | 3600.0000 | 1900.0000 | 1900.0000 |
| early_detection_rate_600_900 | 0.9900 | 0.8870 | 0.0000 | 0.0000 | 0.0000 |
| early_detected | True | True | False | False | False |
| first_detection_time | 603.0000 | 634.0000 | � | 2500.0000 | 2500.0000 |
| mtd_after_end_s | 0.0000 | 0.0000 | 3600.0000 | 1600.0000 | 1600.0000 |

## MLP Autoencoder (Unsupervised FeedForward)

| metric | solar_failure | radiator_failure | normal | solar_holdout | radiator_holdout |
|---|---|---|---|---|---|
| accuracy | 0.9653 | 0.9628 | 0.9658 | 1.0000 | 1.0000 |
| precision | 0.9606 | 0.9605 | 0.0000 | 1.0000 | 1.0000 |
| recall | 0.9993 | 0.9963 | 0.0000 | 1.0000 | 1.0000 |
| f1 | 0.9796 | 0.9781 | 0.0000 | 1.0000 | 1.0000 |
| balanced_accuracy | 0.8972 | 0.8957 | 0.9658 | 1.0000 | 1.0000 |
| mcc | 0.8716 | 0.8615 | 0.0000 | 0.0000 | 0.0000 |
| tn | 477 | 477 | 3477 | 0 | 0 |
| fp | 123 | 123 | 123 | 0 | 0 |
| fn | 2 | 11 | 0 | 0 | 0 |
| tp | 2998 | 2989 | 0 | 1100 | 1100 |
| fpr | 0.2050 | 0.2050 | 0.0342 | 0.0000 | 0.0000 |
| fnr | 0.0007 | 0.0037 | 0.0000 | 0.0000 | 0.0000 |
| roc_auc | 0.9993 | 0.9971 | nan | nan | nan |
| pr_auc | 0.9999 | 0.9994 | nan | nan | nan |
| fpr_before_600 | 0.2050 | 0.2050 | 0.2050 | 0.0000 | 0.0000 |
| tpr_after_900 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| detection_delay_s | 2.0000 | 11.0000 | 3600.0000 | 1900.0000 | 1900.0000 |
| early_detection_rate_600_900 | 0.9934 | 0.9635 | 0.0000 | 0.0000 | 0.0000 |
| early_detected | True | True | False | False | False |
| first_detection_time | 602.0000 | 611.0000 | � | 2500.0000 | 2500.0000 |
| mtd_after_end_s | 0.0000 | 0.0000 | 3600.0000 | 1600.0000 | 1600.0000 |

## Hybrid DIF (Unsupervised Hybrid Deep Isolated Forest)

| metric | solar_failure | radiator_failure | normal | solar_holdout | radiator_holdout |
|---|---|---|---|---|---|
| accuracy | 0.8758 | 0.8539 | 0.8764 | 1.0000 | 1.0000 |
| precision | 0.8708 | 0.8677 | 0.0000 | 1.0000 | 1.0000 |
| recall | 0.9993 | 0.9730 | 0.0000 | 1.0000 | 1.0000 |
| f1 | 0.9306 | 0.9173 | 0.0000 | 1.0000 | 1.0000 |
| balanced_accuracy | 0.6288 | 0.6157 | 0.8764 | 1.0000 | 1.0000 |
| mcc | 0.4702 | 0.3483 | 0.0000 | 0.0000 | 0.0000 |
| tn | 155 | 155 | 3155 | 0 | 0 |
| fp | 445 | 445 | 445 | 0 | 0 |
| fn | 2 | 81 | 0 | 0 | 0 |
| tp | 2998 | 2919 | 0 | 1100 | 1100 |
| fpr | 0.7417 | 0.7417 | 0.1236 | 0.0000 | 0.0000 |
| fnr | 0.0007 | 0.0270 | 0.0000 | 0.0000 | 0.0000 |
| roc_auc | 0.9992 | 0.9649 | nan | nan | nan |
| pr_auc | 0.9999 | 0.9935 | nan | nan | nan |
| fpr_before_600 | 0.7417 | 0.7417 | 0.7417 | 0.0000 | 0.0000 |
| tpr_after_900 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| detection_delay_s | 2.0000 | 81.0000 | 3600.0000 | 1900.0000 | 1900.0000 |
| early_detection_rate_600_900 | 0.9934 | 0.7309 | 0.0000 | 0.0000 | 0.0000 |
| early_detected | True | True | False | False | False |
| first_detection_time | 602.0000 | 681.0000 | � | 2500.0000 | 2500.0000 |
| mtd_after_end_s | 0.0000 | 0.0000 | 3600.0000 | 1600.0000 | 1600.0000 |

## FCNN Supervised (MLP 100-50-20)

| metric | solar_failure | radiator_failure | normal | solar_holdout | radiator_holdout |
|---|---|---|---|---|---|
| accuracy | 0.9989 | 0.9989 | 0.9997 | 1.0000 | 1.0000 |
| precision | 0.9997 | 0.9997 | 0.0000 | 1.0000 | 1.0000 |
| recall | 0.9990 | 0.9990 | 0.0000 | 1.0000 | 1.0000 |
| f1 | 0.9993 | 0.9993 | 0.0000 | 1.0000 | 1.0000 |
| balanced_accuracy | 0.9987 | 0.9987 | 0.9997 | 1.0000 | 1.0000 |
| mcc | 0.9960 | 0.9960 | 0.0000 | 0.0000 | 0.0000 |
| tn | 599 | 599 | 3599 | 0 | 0 |
| fp | 1 | 1 | 1 | 0 | 0 |
| fn | 3 | 3 | 0 | 0 | 0 |
| tp | 2997 | 2997 | 0 | 1100 | 1100 |
| fpr | 0.0017 | 0.0017 | 0.0003 | 0.0000 | 0.0000 |
| fnr | 0.0010 | 0.0010 | 0.0000 | 0.0000 | 0.0000 |
| roc_auc | 1.0000 | 1.0000 | nan | nan | nan |
| pr_auc | 1.0000 | 1.0000 | nan | nan | nan |
| fpr_before_600 | 0.0017 | 0.0017 | 0.0017 | 0.0000 | 0.0000 |
| tpr_after_900 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| detection_delay_s | 3.0000 | 3.0000 | 3600.0000 | 1900.0000 | 1900.0000 |
| early_detection_rate_600_900 | 0.9900 | 0.9900 | 0.0000 | 0.0000 | 0.0000 |
| early_detected | True | True | False | False | False |
| first_detection_time | 603.0000 | 603.0000 | � | 2500.0000 | 2500.0000 |
| mtd_after_end_s | 0.0000 | 0.0000 | 3600.0000 | 1600.0000 | 1600.0000 |

## XGBOD Supervised (Extreme Boosting Outlier Detector)

| metric | solar_failure | radiator_failure | normal | solar_holdout | radiator_holdout |
|---|---|---|---|---|---|
| accuracy | 0.4817 | 0.4825 | 1.0000 | 1.0000 | 1.0000 |
| precision | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| recall | 0.3780 | 0.3790 | 0.0000 | 1.0000 | 1.0000 |
| f1 | 0.5486 | 0.5497 | 0.0000 | 1.0000 | 1.0000 |
| balanced_accuracy | 0.6890 | 0.6895 | 1.0000 | 1.0000 | 1.0000 |
| mcc | 0.3033 | 0.3039 | 0.0000 | 0.0000 | 0.0000 |
| tn | 600 | 600 | 3600 | 0 | 0 |
| fp | 0 | 0 | 0 | 0 | 0 |
| fn | 1866 | 1863 | 0 | 0 | 0 |
| tp | 1134 | 1137 | 0 | 1100 | 1100 |
| fpr | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| fnr | 0.6220 | 0.6210 | 0.0000 | 0.0000 | 0.0000 |
| roc_auc | 0.9959 | 0.9962 | nan | nan | nan |
| pr_auc | 0.9991 | 0.9992 | nan | nan | nan |
| fpr_before_600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tpr_after_900 | 0.4079 | 0.4116 | 0.0000 | 1.0000 | 1.0000 |
| detection_delay_s | 5.0000 | 4.0000 | 3600.0000 | 1900.0000 | 1900.0000 |
| early_detection_rate_600_900 | 0.1096 | 0.0864 | 0.0000 | 0.0000 | 0.0000 |
| early_detected | True | True | False | False | False |
| first_detection_time | 605.0000 | 604.0000 | � | 2500.0000 | 2500.0000 |
| mtd_after_end_s | 0.0000 | 0.0000 | 3600.0000 | 1600.0000 | 1600.0000 |

## Custom Physics-Informed NN (Best)

| metric | solar_failure | radiator_failure | normal | solar_holdout | radiator_holdout |
|---|---|---|---|---|---|
| accuracy | 0.9950 | 0.9936 | 0.9997 | 1.0000 | 1.0000 |
| precision | 0.9997 | 0.9997 | 0.0000 | 1.0000 | 1.0000 |
| recall | 0.9943 | 0.9927 | 0.0000 | 1.0000 | 1.0000 |
| f1 | 0.9970 | 0.9962 | 0.0000 | 1.0000 | 1.0000 |
| balanced_accuracy | 0.9963 | 0.9955 | 0.9997 | 1.0000 | 1.0000 |
| mcc | 0.9823 | 0.9775 | 0.0000 | 0.0000 | 0.0000 |
| tn | 599 | 599 | 3599 | 0 | 0 |
| fp | 1 | 1 | 1 | 0 | 0 |
| fn | 17 | 22 | 0 | 0 | 0 |
| tp | 2983 | 2978 | 0 | 1100 | 1100 |
| fpr | 0.0017 | 0.0017 | 0.0003 | 0.0000 | 0.0000 |
| fnr | 0.0057 | 0.0073 | 0.0000 | 0.0000 | 0.0000 |
| roc_auc | 0.9999 | 0.9999 | nan | nan | nan |
| pr_auc | 1.0000 | 1.0000 | nan | nan | nan |
| fpr_before_600 | 0.0017 | 0.0017 | 0.0017 | 0.0000 | 0.0000 |
| tpr_after_900 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| detection_delay_s | 17.0000 | 22.0000 | 3600.0000 | 1900.0000 | 1900.0000 |
| early_detection_rate_600_900 | 0.9435 | 0.9269 | 0.0000 | 0.0000 | 0.0000 |
| early_detected | True | True | False | False | False |
| first_detection_time | 617.0000 | 622.0000 | � | 2500.0000 | 2500.0000 |
| mtd_after_end_s | 0.0000 | 0.0000 | 3600.0000 | 1600.0000 | 1600.0000 |

