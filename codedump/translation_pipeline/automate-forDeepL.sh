cd data/old-datasets/multilingual-register-data-new/originals
../../../../codedump/translation_pipeline/./format.sh
cd ../formatted
../../../../codedump/translation_pipeline/./shorten_all.sh
cd ../../../preprocessing/preprocessed_texts
../../../codedump/translation_pipeline/./downsample.sh
cd ../../downsampled
rm -rf ../preprocessing/ForDeepL
mkdir -p ../preprocessing/ForDeepL/labels
../../codedump/translation_pipeline/./todocx.sh