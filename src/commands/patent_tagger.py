import sys
sys.path.append("src/")

from preprocessor.patent_data_preprocessor import PatentDataPreprocessor

preprocessor = PatentDataPreprocessor()
preprocessor.start()