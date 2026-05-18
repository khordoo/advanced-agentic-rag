from langgraph.store.base import embed
from pydantic_core.core_schema import FloatSchema
from sqlalchemy import select, desc, func

from rag_core.embeddings.factory import create_embedding_model
from rag_core.store.postgres.models.documents import DocumentChunk
from rag_core.store.postgres.session import get_database

query="Unit volume was unchanged as growth in Latin America (due to market growth)"
embed = create_embedding_model()
db = get_database()
print("Query: ", query)
limit=10
rrf_k=60
with db.get_session() as session:
    query_embed=embed.embed_query(query)
    distance = DocumentChunk.embedding.cosine_distance(query_embed).label('distance')

    stmt2 = (
        select(DocumentChunk,distance).order_by(distance.asc()).limit(limit)
    )
    results = session.execute(stmt2).all()
    print("Simillar results:")
    ss_map={}
    for i, (doc,score) in enumerate(results,start=1):
        print("-" * 20)
        print(f"ID={doc.id} , distance={score:0.3f}, Similairty={1-score:0.3f} TEXT= {doc.text}")
        ss_map[doc.id]=1/(i+rrf_k)

    print('ss ranks: ',ss_map.values())

    print('=========================Text search-----------------')
    query_split= ' OR '.join(query.split(' '))
    print('query split:',query_split)
    ts_query =func.websearch_to_tsquery('english', query_split)
    stmt=(
        select(
            DocumentChunk,
            func.ts_rank_cd(DocumentChunk.fts_vector,ts_query).label('keyword_score')
        )
        .where(DocumentChunk.fts_vector.op("@@")(ts_query))
        .order_by(desc("keyword_score"))
        .limit(limit)
    )
    results = session.execute(stmt).all()
    print("Simillar results:")
    fts_map={}
    for i,(doc,score) in enumerate(results,start=1):
        print("-" * 20)
        print(f"ID={doc.id} , keyword_score={score:0.3f}, TEXT= {doc.text}")
        fts_map[doc.id]=1/(i+rrf_k)

    print('============================RRF===')
    print('--ss:',ss_map)
    print('--fts:',fts_map)
    print('--'*20)
    rrf=[]
    a=set(ss_map.keys())
    b=set(fts_map.keys())
    all_ides= a.union(b)
    print('all ides:',all_ides)
    vector_weight = 0.5
    text_weight = 0.5
    for id in all_ides:
        ss_score=ss_map.get(id,0)
        fte_score = fts_map.get(id,0)
        final_score = text_weight*fte_score+vector_weight*ss_score
        rrf.append((id, final_score ))
    rrf.sort(key=lambda x: x[1],reverse=True)
    print(rrf)

# from sqlalchemy.engine import create_engine
# from sqlalchemy.orm import relationship, sessionmaker
# from sqlalchemy.orm import declarative_base , declarative_base, mapped_column, Mapped
# from sqlalchemy import(
#     INTEGER,String,
#     ForeignKey
# )
# Base = declarative_base()
# class Student(Base):
#     __tablename__='students'
#     id: Mapped[int]= mapped_column(INTEGER,primary_key=True,autoincrement=True)
#     name: Mapped[str]= mapped_column(String(255))
#     classrooms=relationship("ClassRoom", back_populates="student")
#     def __repr__(self) -> str:
#         return f'Student(id={self.id},name={self.name})'

# class ClassRoom(Base):
#     __tablename__='classrooms'
#     id:Mapped[int]= mapped_column(INTEGER,primary_key=True,autoincrement=True)
#     name: Mapped[str]=mapped_column(String(255))
#     student_id: Mapped[int] = mapped_column(ForeignKey(Student.id),nullable=True)
#     student = relationship(Student, back_populates="classrooms")
#     def __repr__(self) -> str:
#         return f'ClassRoom(id={self.id},name={self.name}, student_id={self.student_id})'
# from rag_core.store.postgres.session import get_database
# db = get_database()
# with db.get_session() as session:
#     john = Student(name='John11')
#     session.add(john)
#     session.commit()
#     class_101 = ClassRoom(name='104')
#     class_101.student = john
#     john.classrooms.append(class_101)
#     session.add(john)
#     session.commit()
#     print('john :',john)
# print('class :',class_101)

# url = 'postgresql+psycopg://postgres:postgres@localhost:5432/rag_core'
# engine = create_engine(url)

# Session = sessionmaker(bind=engine)
# session = Session()
# with Session() as session:
#     john = Student(name='John6')
#     # session.add(john)
#     # session.commit()
#     class_101 = ClassRoom(name='class103')
#     # class_101.student = john
#     john.classrooms.append(class_101)
#     session.add(john)
#     session.commit()
#     print('john :',john)
#     print('class :',class_101)
