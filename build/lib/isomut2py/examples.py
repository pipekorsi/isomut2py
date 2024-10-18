import tarfile as __tarfile
import urllib as __urllib
import os as __os
from datetime import datetime as __datetime


def __sizeof_fmt(num, suffix='B'):
    """

    Get size in a human readable object.

    :param num: size in blocks (int)
    :param suffix: suffix to use for output strings (default: "B") (str)

    :returns: string of size (str)

    """
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def __download_file(url, fname):
    """

    Download a file from url to fname.

    :param url: URL to the file to download (str)
    :param fname: path to the desired file (str)

    """
    print('Downloading file from URL "' + url + '" to ' + fname)

    site = __urllib.request.urlopen(url)
    meta = site.info()
    filesize = __sizeof_fmt(int(meta['Content-Length']))
    print('File size: ' + filesize)
    print('Downloading might take a while...')

    try:
        starting_time = __datetime.now()
        __urllib.request.urlretrieve(url, fname)
        finish_time = __datetime.now()
        total_time = finish_time - starting_time
        total_time_h = int(total_time.seconds / 3600)
        total_time_m = int((total_time.seconds % 3600) / 60)
        total_time_s = (total_time.seconds % 3600) % 60
        print('Download completed in ' + str(total_time.days) + ' day(s), ' + str(total_time_h) + ' hour(s), '
              + str(total_time_m) + ' min(s), ' + str(total_time_s) + ' sec(s).')
    except:
        raise ValueError('Downloading failed, try downloading and extracting manually.')

def __extract_file(path, fname):
    """

    Extract tar.gz file to path.

    :param fname: path to the file to extract (str)
    :param path: where to extract (str)

    """
    print('-' * 40)
    print('Extracting downloaded file to ' + path)
    try:
        starting_time = __datetime.now()
        tar = __tarfile.open(fname)
        tar.extractall(path=path)
        tar.close()
        finish_time = __datetime.now()
        total_time = finish_time - starting_time
        total_time_h = int(total_time.seconds / 3600)
        total_time_m = int((total_time.seconds % 3600) / 60)
        total_time_s = (total_time.seconds % 3600) % 60
        print('Extracting completed in ' + str(total_time.days) + ' day(s), ' + str(total_time_h) + ' hour(s), '
              + str(total_time_m) + ' min(s), ' + str(total_time_s) + ' sec(s).')
    except:
        raise ValueError('Extracting failed, try extracting manually.')


def download_example_data(path='.'):
    """

    Download example data from http://genomics.hu/tools/isomut2py/isomut2py_exampleDataset.tar.gz to path.

    :param path: where to download (default: '.') (str)

    """
    url = "http://genomics.hu/tools/isomut2py/isomut2py_example_dataset.tar.gz"

    if path[-1] == '/':
        path = path[:-1]
    if not __os.path.isdir(path):
        __os.mkdir(path)

    fname = path + '/' + url.split('/')[-1]

    __download_file(url=url, fname=fname)
    __extract_file(path=path, fname=fname)


def download_raw_example_data(path='.'):
    """

    Download raw example data from either http://genomics.hu/tools/isomut2py/isomut2py_rawExampleDataset.tar.gz or http://genomics.hu/tools/isomut2py/isomut2py_rawExampleDataset_shortgenome.tar.gz to path.

    :param path: where to download (default: '.') (str)

    """
    if path[-1] == '/':
        path = path[:-1]
    if not __os.path.isdir(path):
        __os.mkdir(path)

    url = "http://genomics.hu/tools/isomut2py/isomut2py_raw_example_dataset.tar.gz"

    fname = path + '/' + url.split('/')[-1]

    __download_file(url=url, fname=fname)
    __extract_file(path=path, fname=fname)

def load_example_mutdet_settings(example_data_path='.', output_dir='.'):
    """

    Loads example settings for mutation detection.

    :param example_data_path: the path where example data is located (default: '.') (str)
    :param output_dir: the path where results should be saved (default: current working directory) (str)

    :returns: dictionary containing example parameters

    """
    if example_data_path[-1] == '/':
        example_data_path = example_data_path[:-1]
    example_data_path += '/isomut2py_example_dataset/MutationDetection'

    exampleParams = dict()
    exampleParams['input_dir'] = example_data_path + '/alignmentFiles/'
    exampleParams['bam_filename'] = ['S01.bam', 'S02.bam', 'S03.bam', 'S04.bam', 'S05.bam', 'S06.bam', 'S07.bam',
                                     'S08.bam', 'S09.bam', 'S10.bam']
    exampleParams['output_dir'] = output_dir
    exampleParams['ref_fasta'] = example_data_path + '/referenceGenome/refgenome.fa'
    exampleParams['chromosomes'] = ['1']
    exampleParams['n_min_block'] = 100

    return exampleParams

def load_example_ploidyest_settings(example_data_path='.', output_dir='.'):
    """

    Loads example settings for ploidy estimation.

    :param example_data_path: the path where example data is located (default: '.') (str)
    :param output_dir: the path where results should be saved (default: current working directory) (str)

    :returns: dictionary containing example parameters

    """
    if example_data_path[-1] == '/':
        example_data_path = example_data_path[:-1]

    example_data_path += '/isomut2py_raw_example_dataset/'
    exampleParams = dict()
    exampleParams['input_dir'] = example_data_path
    exampleParams['bam_filename'] = 'A.bam'
    exampleParams['output_dir'] = output_dir
    exampleParams['ref_fasta'] = example_data_path + 'refgenome.fa'
    exampleParams['n_min_block'] = 100
    exampleParams['n_conc_blocks'] = 4
    exampleParams['windowsize'] = 200
    exampleParams['shiftsize'] = 150
    exampleParams['min_noise'] = 0
    exampleParams['base_quality_limit'] = 30
    exampleParams['windowsize_PE'] = 300
    exampleParams['shiftsize_PE'] = 200
    exampleParams['print_every_nth'] = 100
    exampleParams['cov_max'] = 2000
    exampleParams['cov_min'] = 5
    exampleParams['hc_percentile'] = 90
    exampleParams['chromosomes'] = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII',
                               'XIV', 'XV', 'XVI']

    exampleParams['chrom_length'] = [230218, 813184, 316620, 1531933, 576874, 270161, 1090940, 562643, 439888, 745751,
                                666816,
                                1078177, 924431, 784333, 1091291, 948066]  # length of chroms (ensembl.org)

    return exampleParams

def load_preprocessed_example_ploidyest_settings(example_data_path='.'):
    """

    Loads example settings for ploidy estimation with preprocessed files.

    :param example_data_path: the path where example data is located (default: '.') (str)

    :returns: dictionary containing example parameters

    """
    if example_data_path[-1] == '/':
        example_data_path = example_data_path[:-1]
    example_data_path += '/isomut2py_example_dataset/PloidyEstimation'

    exampleParams = dict()
    exampleParams['input_dir'] = ''
    exampleParams['bam_filename'] = 'S01.bam'
    exampleParams['output_dir'] = example_data_path+'/'
    exampleParams['ref_fasta'] = ''
    exampleParams["chromosomes"] = [str(k) for k in range(1, 23)] + ['X', 'Y']
    exampleParams['chrom_length'] = [248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973, 145138636,
                            138394717, 133797422, 135086622, 133275309, 114364328, 107043718, 101991189, 90338345,
                            83257441, 80373285, 58617616, 64444167, 46709983,
                            50818468, 156040895, 57227415]
    exampleParams['genome_length'] = sum(exampleParams['chrom_length'])

    return exampleParams