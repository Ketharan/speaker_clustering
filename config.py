# Flask settings
DEBUG = False

# Flask-restplus settings
RESTPLUS_MASK_SWAGGER = False

# Application settings

# API metadata
API_TITLE = 'MAX Audio Embedding Generator'
API_DESC = 'Generate embedding vectors from audio files.'
API_VERSION = '1.1.0'



DEFAULT_EMBEDDING_CHECKPOINT = "common/data/experiments/nets/assets/vggish_model.ckpt"
DEFAULT_PCA_PARAMS = "common/data/experiments/nets/assets/vggish_pca_params.npz"
AUDIO_DATA_DIRECTORY = "/home/ketharan/ZHAW_deep_voice/demo/api_audio"
VECTOR_SIZE = 128



'''
wget -nv --show-progress --progress=bar:force:noscroll http://max-assets.s3.us.cloud-object-storage.appdomain.cloud/audio-embedding-generat
or/1.0/assets.tar.gzoad image 


'''
