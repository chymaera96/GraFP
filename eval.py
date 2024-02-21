# This code is sourced from https://github.com/mimbres/neural-audio-fp.git

import faiss
import time
import numpy as np
import os

def get_index(index_type,
              train_data,
              train_data_shape,
              use_gpu=True,
              max_nitem_train=2e7,
              n_centroids=64,
):
    """
    • Create FAISS index
    • Train index using (partial) data
    • Return index
    Parameters
    ----------
    index_type : (str)
        Index type must be one of {'L2', 'IVF', 'IVFPQ', 'IVFPQ-RR',
                                   'IVFPQ-ONDISK', HNSW'}
    train_data : (float32)
        numpy.memmap or numpy.ndarray
    train_data_shape : list(int, int)
        Data shape (n, d). n is the number of items. d is dimension.
    use_gpu: (bool)
        If False, use CPU. Default is True.
    max_nitem_train : (int)
        Max number of items to be used for training index. Default is 1e7.
    Returns
    -------
    index : (faiss.swigfaiss_avx2.GpuIndex***)
        Trained FAISS index.
    References:
        https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
    """
    # GPU Setup
    if use_gpu:
        GPU_RESOURCES = faiss.StandardGpuResources()
        GPU_OPTIONS = faiss.GpuClonerOptions()
        GPU_OPTIONS.useFloat16 = True # use float16 table to avoid https://github.com/facebookresearch/faiss/issues/1178
        #GPU_OPTIONS.usePrecomputed = False
        #GPU_OPTIONS.indicesOptions = faiss.INDICES_CPU
    else:
        pass

    # Fingerprint dimension, d
    d = train_data_shape[1]

    # Build a flat (CPU) index
    index = faiss.IndexFlatL2(d) #

    mode = index_type.lower()
    print(f'Creating index: \033[93m{mode}\033[0m')
    if mode == 'l2':
        # Using L2 index
        pass
    elif mode == 'ivf':
        # Using IVF index
        nlist = 400
        index = faiss.IndexIVFFlat(index, d, nlist)
    elif mode == 'ivfpq':
        # Using IVF-PQ index
        code_sz = 64 # power of 2
        nbits = 8  # nbits must be 8, 12 or 16, The dimension d should be a multiple of M.
        index = faiss.IndexIVFPQ(index, d, n_centroids, code_sz, nbits)
    elif mode == 'ivfpq-rr':
        # Using IVF-PQ index + Re-rank
        code_sz = 64
        # n_centroids = 256 # 10:1.92ms, 30:1.29ms, 100: 0.625ms
        nbits = 8  # nbits must be 8, 12 or 16, The dimension d should be a multiple of M.
        M_refine = 4
        nbits_refine = 4
        index = faiss.IndexIVFPQR(index, d, n_centroids, code_sz, nbits,
                                  M_refine, nbits_refine)
    elif mode == 'ivfpq-ondisk':
        if use_gpu:
            raise NotImplementedError(f'{mode} is only available in CPU.')
        raise NotImplementedError(mode)
    elif mode == 'hnsw':
        if use_gpu:
            raise NotImplementedError(f'{mode} is only available in CPU.')
        else:
            M = 16
            index = faiss.IndexHNSWFlat(d, M)
            index.hnsw.efConstruction = 80
            index.verbose = True
            index.hnsw.search_bounded_queue = True
    else:
        raise ValueError(mode.lower())

    # From CPU index to GPU index
    if use_gpu:
        print('Copy index to \033[93mGPU\033[0m.')
        index = faiss.index_cpu_to_gpu(GPU_RESOURCES, 0, index, GPU_OPTIONS)

    # Train index
    start_time = time.time()
    if len(train_data) > max_nitem_train:
        print('Training index using {:>3.2f} % of data...'.format(
            100. * max_nitem_train / len(train_data)))
        # shuffle and reduce training data
        sel_tr_idx = np.random.permutation(len(train_data))
        sel_tr_idx = sel_tr_idx[:max_nitem_train]
        index.train(train_data[sel_tr_idx,:])
    else:
        print('Training index...')
        index.train(train_data) # Actually do nothing for {'l2', 'hnsw'}
    print('Elapsed time: {:.2f} seconds.'.format(time.time() - start_time))

    # N probe
    index.nprobe = 20
    return index


def load_memmap_data(source_dir,
                     fname,
                     append_extra_length=None,
                     shape_only=False,
                     display=True):
    """
    Load data and datashape from the file path.
    • Get shape from [source_dir/fname_shape.npy}.
    • Load memmap data from [source_dir/fname.mm].
    Parameters
    ----------
    source_dir : (str)
    fname : (str)
        File name except extension.
    append_empty_length : None or (int)
        Length to appened empty vector when loading memmap. If activate, the
        file will be opened as 'r+' mode.
    shape_only : (bool), optional
        Return only shape. The default is False.
    display : (bool), optional
        The default is True.
    Returns
    -------
    (data, data_shape)
    """
    path_shape = os.path.join(source_dir, fname + '_shape.npy')
    path_data = os.path.join(source_dir, fname + '.mm')
    data_shape = np.load(path_shape)
    if shape_only:
        return data_shape

    if append_extra_length:
        data_shape[0] += append_extra_length
        data = np.memmap(path_data, dtype='float32', mode='r+',
                         shape=(data_shape[0], data_shape[1]))
    else:
        data = np.memmap(path_data, dtype='float32', mode='r',
                         shape=(data_shape[0], data_shape[1]))
    if display:
        print(f'Load {data_shape[0]:,} items from \033[32m{path_data}\033[0m.')
    return data, data_shape

def eval_faiss(emb_dir,
               emb_dummy_dir=None,
               index_type='ivfpq',
               nogpu=False,
               max_train=1e7,
               test_ids='icassp',
               test_seq_len='1 3 5 9 11 19',
               k_probe=20,
               n_centroids=64):
    """
    Segment/sequence-wise audio search experiment and evaluation: implementation based on FAISS.
    """
    if type(test_seq_len) == str:
        test_seq_len = np.asarray(
            list(map(int, test_seq_len.split())))  # '1 3 5' --> [1, 3, 5]

    # Load items from {query, db, dummy_db}
    query, query_shape = load_memmap_data(emb_dir, 'query')
    db, db_shape = load_memmap_data(emb_dir, 'db')
    if emb_dummy_dir is None:
        emb_dummy_dir = emb_dir
    dummy_db, dummy_db_shape = load_memmap_data(emb_dummy_dir, 'dummy_db')
    """ ----------------------------------------------------------------------
    FAISS index setup
        dummy: 10 items.
        db: 5 items.
        query: 5 items, corresponding to 'db'.
        index.add(dummy_db); index.add(db) # 'dummy_db' first
               |------ dummy_db ------|
        index: [d0, d1, d2,..., d8, d9, d11, d12, d13, d14, d15]
                                       |--------- db ----------|
                                       |--------query ---------|
                                       [q0,  q1,  q2,  q3,  q4]
    • The set of ground truth IDs for q[i] will be (i + len(dummy_db))
    ---------------------------------------------------------------------- """
    # Create and train FAISS index
    index = get_index(index_type, dummy_db, dummy_db.shape, (not nogpu),
                      max_train, n_centroids=n_centroids)

    # Add items to index
    start_time = time.time()

    index.add(dummy_db); print(f'{len(dummy_db)} items from dummy DB')
    index.add(db); print(f'{len(db)} items from reference DB')

    t = time.time() - start_time
    print(f'Added total {index.ntotal} items to DB. {t:>4.2f} sec.')

    """ ----------------------------------------------------------------------
    We need to prepare a merged {dummy_db + db} memmap:
    • Calcuation of sequence-level matching score requires reconstruction of
      vectors from FAISS index.
    • Unforunately, current faiss.index.reconstruct_n(id_start, id_stop)
      supports only CPU index.
    • We prepare a fake_recon_index thourgh the on-disk method.
    ---------------------------------------------------------------------- """
    # Prepare fake_recon_index
    del dummy_db
    start_time = time.time()

    fake_recon_index, index_shape = load_memmap_data(
        emb_dummy_dir, 'dummy_db', append_extra_length=query_shape[0],
        display=False)
    fake_recon_index[dummy_db_shape[0]:dummy_db_shape[0] + query_shape[0], :] = db[:, :]
    fake_recon_index.flush()

    t = time.time() - start_time
    print(f'Created fake_recon_index, total {index_shape[0]} items. {t:>4.2f} sec.')

    # Get test_ids
    print(f'test_id: \033[93m{test_ids}\033[0m,  ', end='')
    if test_ids.lower() == 'all':
        test_ids = np.arange(0, len(query) - max(test_seq_len), 1) # will test all segments in query/db set
    elif test_ids.isnumeric():
        test_ids = np.random.permutation(len(query) - max(test_seq_len))[:int(test_ids)]
    else:
        test_ids = np.load(test_ids)

    n_test = len(test_ids)
    gt_ids  = test_ids + dummy_db_shape[0]
    print(f'n_test: \033[93m{n_test:n}\033[0m')

    """ Segement/sequence-level search & evaluation """
    # Define metric
    top1_exact = np.zeros((n_test, len(test_seq_len))).astype(int) # (n_test, test_seg_len)
    top1_near = np.zeros((n_test, len(test_seq_len))).astype(int)
    top3_exact = np.zeros((n_test, len(test_seq_len))).astype(int)
    top10_exact = np.zeros((n_test, len(test_seq_len))).astype(int)
    # top1_song = np.zeros((n_test, len(test_seq_len))).astype(np.int)

    start_time = time.time()
    for ti, test_id in enumerate(test_ids):
        gt_id = gt_ids[ti]
        for si, sl in enumerate(test_seq_len):
            assert test_id <= len(query)
            q = query[test_id:(test_id + sl), :] # shape(q) = (length, dim)

            # segment-level top k search for each segment
            _, I = index.search(
                q, k_probe) # _: distance, I: result IDs matrix

            # offset compensation to get the start IDs of candidate sequences
            for offset in range(len(I)):
                I[offset, :] -= offset

            # unique candidates
            candidates = np.unique(I[np.where(I >= 0)])   # ignore id < 0

            """ Sequence match score """
            _scores = np.zeros(len(candidates))
            for ci, cid in enumerate(candidates):
                _scores[ci] = np.mean(
                    np.diag(
                        # np.dot(q, index.reconstruct_n(cid, (cid + l)).T)
                        np.dot(q, fake_recon_index[cid:cid + sl, :].T)
                        )
                    )

            """ Evaluate """
            pred_ids = candidates[np.argsort(-_scores)[:10]]
            # pred_id = candidates[np.argmax(_scores)] <-- only top1-hit

            # top1 hit
            top1_exact[ti, si] = int(gt_id == pred_ids[0])
            top1_near[ti, si] = int(
                pred_ids[0] in [gt_id - 1, gt_id, gt_id + 1])
            # top1_song = need song info here...

            # top3, top10 hit
            top3_exact[ti, si] = int(gt_id in pred_ids[:3])
            top10_exact[ti, si] = int(gt_id in pred_ids[:10])


    # Summary
    top1_exact_rate = 100. * np.mean(top1_exact, axis=0)
    top1_near_rate = 100. * np.mean(top1_near, axis=0)
    top3_exact_rate = 100. * np.mean(top3_exact, axis=0)
    top10_exact_rate = 100. * np.mean(top10_exact, axis=0)
    # top1_song = 100 * np.mean(top1_song[:ti + 1, :], axis=0)

    hit_rates = np.stack([top1_exact_rate, top1_near_rate, top3_exact_rate, top10_exact_rate])
    del fake_recon_index, query, db

    # print(hit_rates)
    np.save(f'{emb_dir}/hit_rates.npy', hit_rates)

    np.save(f'{emb_dir}/raw_score.npy',
            np.concatenate(
                (top1_exact, top1_near, top3_exact, top10_exact), axis=1))
    np.save(f'{emb_dir}/test_ids.npy', test_ids)
    print(f'Saved test_ids, hit-rates and raw score to {emb_dir}.')

    return hit_rates

