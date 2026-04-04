from prefect import flow
from fsi_credit.pipelines.ingest_to_bronze import main as bronze
from fsi_credit.pipelines.bronze_to_silver import main as silver
from fsi_credit.pipelines.silver_to_gold import main as gold

@flow
def medalion_flow():
    bronze()
    silver()
    gold()
