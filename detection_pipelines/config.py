CONDITIONS = ["CalendarExtraction", "statistics"]
COND_FEATURES = 4
DATASETS = [

  # ("y1", "../data/in_train_ID200.csv", "2011-01-01 00:15:00", "time", 24 * 4, "15min", True,
  #  "../data/out_class_1_small.csv", "Small_y1"),
  # ("y2", "../data/in_train_ID200.csv", "2011-01-01 00:15:00", "time", 24 * 4, "15min", True,
  #  "../data/out_class_2_small.csv", "Small_y2"),
  # ("y3", "../data/in_train_ID200.csv", "2011-01-01 00:15:00", "time", 24 * 4, "15min", True,
  #  "../data/out_class_3_small.csv", "Small_y3"),
  # ("y4", "../data/in_train_ID200.csv", "2011-01-01 00:15:00", "time", 24 * 4, "15min", True,
  #  "../data/out_class_4_small.csv", "Small_y4"),
  #      ("y", "../data/in_train_ID200.csv", "2011-01-01 00:15:00", "time", 24 * 4, "15min", True,
  #  "../data/out_train_ID200_50_50_50_50_small.csv", "Small"),

     

    ("y", "../data/in_train_ID200.csv", "2011-01-01 00:15:00", "time", 24 * 4, "15min", True,
       "../data/out_train_ID200_25_25_25_25_small.csv", "Small"),
]
TRAINING_LENGTH = 15000
GS = "optimal" # "default", "optimal", "search"
