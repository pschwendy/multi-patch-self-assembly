{% extends "bridges2.sh" %}

{% block header %}
{{ super() -}}
#SBATCH -o output-files/slurm-%j.out

echo "Running on `hostname`"

module load anaconda3
module load openmpi/4.0.2-clang2.1


{% endblock header %}

{% block body %}
export CT=`date +%s`
export WT=7200
export HOOMD_WALLTIME_STOP=`echo "$CT+$WT-60" | bc`
{{ super() -}}
{% endblock %}
