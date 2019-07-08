import os
import shutil

def find_newrunID(outdir):
    run_names = [fname for fname in os.listdir(outdir) if "run" in fname]
    #possible awkward bug: if you put a file inside outdir that is not
    #of the form "run" followed by a 3 digit number

    lastrunID = -1
    if len(run_names) > 0:
        run_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        lastrunID = int(''.join(filter(str.isdigit, run_names[-1])))

    return lastrunID + 1 #newrunID

def resolve_run_directory(args):
    # create new run directory if -newrun flag specified
    if args.newrun:
        newrunID = find_newrunID(args.outdir)
        os.mkdir(os.path.join(args.outdir, 'run%03d'%newrunID))
        args.ckptdir = os.path.join(args.outdir, 'run%03d'%newrunID, 'checkpoints/')
        args.resultdir = os.path.join(args.outdir, 'run%03d'%newrunID, 'results/')
        os.mkdir(args.ckptdir)
        os.mkdir(args.resultdir)
        #TODO: copy over gin config file
        #TODO: save git commit hash in result directory as well

    if not args.newrun and (args.ckptdir is None or args.resultdir is None):
        raise ValueError("Please either specify the -newrun flag "
                "or provide paths for both --ckptdir and --resultdir")
    return args

def resolve_npy_directory(args):
    if args.newnpy:
        for split_type in ['train', 'validation', 'test']:
            try:
                print("Removing {}".format((os.path.join(args.datadir, split_type))))
                if os.path.isdir(os.path.join(args.datadir, split_type)):
                    shutil.rmtree(os.path.join(args.datadir, split_type))
            except:
                print("Error Occur when removing old numpy directory -- {}".format(os.path.join(args.datadir, split_type)))

        # FIXME: refactor args.pklfile
        if not os.path.isfile(os.path.join(args.datadir, args.pklfile)):
            raise ValueError("Cannot Find the raw pickle file -- {}".format(os.path.join(args.datadir, args.pklfile)))

    return args