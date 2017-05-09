import sys


def train_info(epoch, loss, num_batches=None, updates=None,  time_per_update=None):
    print >> sys.stderr, 'Epoch: ', epoch,
    if num_batches:
        print >> sys.stderr, 'Batches: ', num_batches,
    if updates:
        print >> sys.stderr, 'Updates: ', updates,
    
    print >> sys.stderr, 'Avg.Loss/Batch: ', "{0:.4f}".format(loss/num_batches),
    
    if time_per_update:
        print >> sys.stderr, 'Duration/Update:', time_per_update
    else:
        print >> sys.stderr, ''

def validation_info(validator, print_loss=True, print_metric=False, run_script=False):
    print >> sys.stderr, ''
    print >> sys.stderr, 'Validation...'
    
    if print_loss:
        validation_loss = validator.compute_loss()
        print >> sys.stderr, '\tLoss:', "{0:.4f}".format(validation_loss),
    if print_metric:
        validation_score = validator.compute_metric()
        if validation_score:
            print >> sys.stderr, validator.metric.metric_name, "{0:.4f}".format(validation_score),
    else:
        print >> sys.stderr, ''
    print >> sys.stderr, ''
    if run_script:
        validator.run_external_script()