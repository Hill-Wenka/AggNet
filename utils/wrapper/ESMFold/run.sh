export MKL_THREADING_LAYER=GNU
source "$8"/bin/activate "$9"
echo "using conda environment: $9"
echo "shell script: $0"
echo "python script: $1"
echo "-i: $2"
echo "-o: $3"
echo "--max-tokens-per-batch: $4"
echo "--num-recycles: $5"
echo "--cpu-only: $6"
echo "--cpu-offload: $7"

if [[ $6 == False && $7 == False ]]; then
  python $1 -i $2 -o $3 --max-tokens-per-batch $4 --num-recycles $5
elif [[ $6 == True && $7 == False ]]; then
  python $1 -i $2 -o $3 --max-tokens-per-batch $4 --num-recycles $5 --cpu-only
elif [[ $6 = False && $7 = True ]]; then
  python $1 -i $2 -o $3 --max-tokens-per-batch $4 --num-recycles $5 --cpu-offload
else
  python $1 -i $2 -o $3 --max-tokens-per-batch $4 --num-recycles $5 --cpu-only --cpu-offload
fi
