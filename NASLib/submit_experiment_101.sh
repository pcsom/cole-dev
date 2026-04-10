for i in {10500..10509}
do
  sbatch run_101.sh $i false mlp 5
done