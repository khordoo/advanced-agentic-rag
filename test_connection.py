import time
from sqlalchemy import text, select
from rag_core.store.postgres.repositories.source_assets import SourceAssetRepository
# from rag_core.store.postgres.repositories.pipeline_runs import IngestionRunRepository
from rag_core.store.postgres.models.ingestion import SourceAsset,AssetState
from rag_core.store.postgres.session import get_database
with get_database().get_session() as session:
#     repo = SourceAssetRepository(session)
#     print(repo.get_next_pending())
      # repo=IngestionRunRepository(session)
      # run=repo.create_run()
      # print('scannig new run:',run)
      # time.sleep(10)
      # repo.mark_ingesting(run.id)
      # print('set ingetino to ingestion:',run)
      stmt =(
          select(SourceAsset)
          .where(SourceAsset.state==AssetState.PROCESSING)
      )
      files = session.execute(stmt).scalars()
      print('type of files:',type(files))
      for f in files:
          print(f)
          print('--'*20)
