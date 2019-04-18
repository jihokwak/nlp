from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

data = pd.DataFrame(data=[[0.1, 0.5], [0.1, 0.7], [0.5, 0.95], [0.55, 1.2]], index=['스시', '일본','김치', '한국'])

pd.DataFrame(np.round(cosine_similarity(data, data),3), columns=data.index, index=data.index)



np.dot(data.iloc[0], data.iloc[1])/(np.linalg.norm(data.iloc[0])*np.linalg.norm(data.iloc[1]))

np.dot(data.iloc[0], data.iloc[1])
