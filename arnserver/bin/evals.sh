CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $CUR_DIR/set_env.sh


START=$(date +%s.%N)

expname=pacrrpub
train_years=wt09_10
test_year=wt11_12
numneg=6
batch=32
winlen=1
kmaxpool=3
distill="firstk"
binmat=False
context=False
cascade=''
combine=16
qproximity=0
iterations=10
shuffle=False
parentdir=/home/ubuntu/copacrr
outdir=$parentdir


# $evalf could be 
# evals.docpairs  evals.rerank  evals.stat_latex  evals.summarize
'''
python -m evals.docpairs with\
        expname=$expname \
        train_years=$train_years \
        numneg=$numneg \
        batch=$batch \
        winlen=$winlen \
        kmaxpool=$kmaxpool \
        binmat=$binmat \
        context=$context \
        combine=$combine \
        shuffle=$shuffle \
	cascade=$cascade \
	qproximity=$qproximity \
	parentdir=$parentdir \
        epochs=$iterations \
        outdir=$outdir
'''
python -m evals.rerank with\
        expname=$expname \
        train_years=$train_years \
        numneg=$numneg \
        batch=$batch \
        winlen=$winlen \
        kmaxpool=$kmaxpool \
        binmat=$binmat \
        context=$context \
        combine=$combine \
        shuffle=$shuffle \
	cascade=$cascade \
	qproximity=$qproximity \
	parentdir=$parentdir \
        epochs=$iterations \
        outdir=$outdir
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo finished within $DIFF
