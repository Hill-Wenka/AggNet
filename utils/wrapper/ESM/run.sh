export MKL_THREADING_LAYER=GNU
source "$8"/bin/activate "$9"
echo "using conda environment: $9"
echo "shell script: $0"
echo "python script: $1"
echo "model: $2"
echo "fasta: $3"
echo "output: $4"
repr_layers=(${5//,/ })
echo "repr_layers: ${repr_layers[@]}"
inclue=(${6//,/ })
echo "inclue: ${inclue[@]}"
npgpu=(${7//,/ })
echo "nogpu: ${npgpu[@]}"
if [ "$7" = "True" ]; then
  python $1 $2 $3 $4 --repr_layers ${repr_layers[@]} --include ${inclue[@]} --nogpu
else
  python $1 $2 $3 $4 --repr_layers ${repr_layers[@]} --include ${inclue[@]}
fi
