#######1. tok
perl ./mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < en_cv.en > en_cv.en.tok
perl ./mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < en_cv.cv > en_cv.cv.tok

#######2. bpe

subword-nmt learn-bpe -s 30000 < en_cv.en.tok > en_cv.en.bpe.codes
subword-nmt learn-bpe -s 30000 < en_cv.cv.tok > en_cv.cv.bpe.codes


subword-nmt apply-bpe -c en_cv.en.bpe.codes < en_cv.en.tok > en_cv.en.tok.bpe
subword-nmt apply-bpe -c en_cv.cv.bpe.codes < en_cv.cv.tok > en_cv.cv.tok.bpe


#######3. get vocab

cat en_cv.en.tok.bpe | subword-nmt get-vocab | awk -F ' ' '{print $1}' > vocab.en
cat en_cv.cv.tok.bpe | subword-nmt get-vocab | awk -F ' ' '{print $1}' > vocab.cv

sed -i '1i\</s>' vocab.en
sed -i '1i\<s>' vocab.en
sed -i '1i\<unk>' vocab.en

sed -i '1i\</s>' vocab.cv
sed -i '1i\<s>' vocab.cv
sed -i '1i\<unk>' vocab.cv




