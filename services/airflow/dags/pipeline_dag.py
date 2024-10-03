from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 9, 30),
}

dvc_path = ' /.venv/bin/dvc'
# docker_compose_path = '/mnt/d/PythonProjects/deployment_assignment/docker-compose'

with DAG('pipeline', default_args=default_args, schedule_interval='*/5 * * * *') as dag:
    preprocess = BashOperator(
        task_id='preprocess',
        bash_command=f'cd /mnt/d/PythonProjects/deployment_assignment && dvc repro preprocess'
    )

    train = BashOperator(
        task_id='train',
        bash_command=f'cd /mnt/d/PythonProjects/deployment_assignment && dvc repro train'
    )
    down = BashOperator(
        task_id='down',
        bash_command=f'cd /mnt/d/PythonProjects/deployment_assignment && docker compose -f code/deployment/docker-compose.yml down'
    )
    deploy = BashOperator(
        task_id='deploy',
        bash_command=f'cd /mnt/d/PythonProjects/deployment_assignment && docker compose -f code/deployment/docker-compose.yml up --build -d'
    )
    mlflow_ui = BashOperator(
    	task_id='mlflow',
    	bash_command='fuser -k 5000/tcp; mlflow ui --host 0.0.0.0 --port 5000 > /dev/null 2>&1 &',
    dag=dag,
)

    preprocess >> train >> down >> deploy >> mlflow_ui
