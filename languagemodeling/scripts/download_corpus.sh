tmpname="`echo $(date) | sed 's/[ :]//g'`"
filename="movie_dialogs.corpus"
corpus_path="./corpus"

if [ ! -d $corpus_path ]; then
    echo "$corpus_path directory doesn't exist."
    exit 1
fi

mkdir -p dtmp; cd dtmp
wget http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip
unzip cornell_movie_dialogs_corpus.zip
cd cornell\ movie-dialogs\ corpus
awk -F " \+\+\+.\+\+\+ " '{print $5}' movie_lines.txt > ../../$corpus_path/$tmpname.corpus
cd ../..

# Remove not utf-8 characters from file
iconv -f utf-8 -t utf-8 -c $corpus_path/$tmpname.corpus > $corpus_path/$filename
echo "New file created: ./corpus/$filename"

rm -r dtmp
rm $corpus_path/$tmpname.corpus
