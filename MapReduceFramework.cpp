#include <atomic>
#include <csignal>
#include <condition_variable>
#include <algorithm>
#include <malloc.h>
#include "MapReduceFramework.h"
#include "Barrier.h"

struct JobContext { // resources used by all threads - every thread hold a pointer
    std::vector<IntermediateVec *> *all_intermediate_vec;
    std::vector<IntermediateVec *> *shuffled_intermediate_vec;
    JobState *job_state;
    const MapReduceClient *client;
    OutputVec *output_vec;
    std::mutex* end_map_stage_mutex; //todo: fix
    std::atomic<int> *map_atomic_counter;
    std::atomic<int> *reduce_atomic_counter;
    Barrier *barrier;
};

struct ThreadContext {
    int thread_id;
    const InputVec *input_vec;
    JobContext *job_context; // JobContext of all threads
    IntermediateVec *intermediate_vec;
};

JobState *get_new_job_state();

ThreadContext *init_thread_contexts(OutputVec &outputVec, int multiThreadLevel, const MapReduceClient &client,
                                    const InputVec &inputVec, JobState *job_state, JobContext * job_context);

bool compare(const IntermediatePair &a, const IntermediatePair &b) {
    return *(a.first) < *(b.first);
}

K2 *get_max_key(std::vector<IntermediateVec *> *all_vectors);

void
pop_all_max_keys(K2 *max_key, IntermediateVec *intermediateVecOutput, std::vector<IntermediateVec *> *all_vec_input);

void *map_reduce_method(void *context) {
    auto *tc = (ThreadContext *) context;

    //Map
    tc->job_context->job_state->stage = MAP_STAGE;
    int current_index = (*(tc->job_context->map_atomic_counter))++;
    while (current_index < tc->input_vec->size()) {
        //Todo: how should I increase the percentage
        tc->job_context->client->map(tc->input_vec->at(current_index).first, tc->input_vec->at(current_index).second,
                                     context);
        current_index = (*(tc->job_context->map_atomic_counter))++;
    }
    //Sort
    std::sort(tc->intermediate_vec->begin(), tc->intermediate_vec->end(), compare);
    tc->job_context->barrier->barrier();

    //Shuffle
    tc->job_context->job_state->stage = SHUFFLE_STAGE;
    if (tc->thread_id == 0) {
        std::vector<IntermediateVec *> *all_intermediate_vec = tc->job_context->all_intermediate_vec;
        std::vector<IntermediateVec *> *shuffled_vector = tc->job_context->shuffled_intermediate_vec;

        while (!all_intermediate_vec->empty()) {
            K2 *max_key = get_max_key(all_intermediate_vec); // gets the maximal key of all keys imn all vec
            auto *max_key_vector = (IntermediateVec *) malloc(sizeof(IntermediateVec));
            pop_all_max_keys(max_key, max_key_vector, all_intermediate_vec);
            shuffled_vector->push_back(max_key_vector);
        }

    }
    tc->job_context->barrier->barrier();
    tc->job_context->job_state->stage = REDUCE_STAGE;
    //Reduce
    current_index = (*(tc->job_context->reduce_atomic_counter))++;
    while (current_index < tc->job_context->shuffled_intermediate_vec->size()) {
        tc->job_context->client->reduce(tc->job_context->shuffled_intermediate_vec->at(current_index), context);
        current_index = (*(tc->job_context->reduce_atomic_counter))++;
    }
    return tc->job_context->job_state;
}

void remove_empty_vectors(std::vector<IntermediateVec *> *intermediate_vec);

void
pop_all_max_keys(K2 *max_key, IntermediateVec *intermediateVecOutput, std::vector<IntermediateVec *> *all_vec_input) {
    for (auto vec: *all_vec_input) {
        while (vec->back().first == max_key) {
            intermediateVecOutput->push_back(vec->back());
            vec->pop_back();
        }
    }
    //after finishing emptying every vector from max_key pairs - going over and deleting all empty vectors
    remove_empty_vectors(all_vec_input);
}


bool free_if_empty(const IntermediateVec *intermediate_vec) {
    //frees the vec alloc when empty, and returns true. otherwise false without free
    bool flag = intermediate_vec->empty();
    if (flag) {
        delete intermediate_vec;
    }
    return flag;
}

void remove_empty_vectors(std::vector<IntermediateVec *> *all_vectors) {
    // Remove empty vectors
    all_vectors->erase(std::remove_if(all_vectors->begin(), all_vectors->end(),
                                      [](const IntermediateVec *intermediate_vec) {
                                          return free_if_empty(intermediate_vec);
                                      }), all_vectors->end());
}

K2 *get_max_key(std::vector<IntermediateVec *> *all_vectors) {
    K2 *max_key = (K2 *) malloc(sizeof(K2));
    bool is_first_iteration = true;
    for (auto vec: *all_vectors) {
        if (is_first_iteration) {
            max_key = vec->back().first;
            is_first_iteration = false;
            continue;
        }
        max_key = std::max(max_key, vec->back().first);
    }
    return max_key;
}

JobContext *init_job_context(OutputVec &outputVec, const MapReduceClient &client, JobState *job_state,
//                             std::atomic<int> *map_atomic_counter,std::atomic<int> *reduce_atomic_counter,
//                             std::vector<IntermediateVec *> *all_vectors,
//                             std::vector<IntermediateVec *> *shuffled_intermediate_vec,
                             int numOfThreads);

/**
 * starts running the MapReduce algorithm (w several threads) and returns a job handler
 * You can assume that the input to this function is valid
 * @param client the task that the framework should run (The implementation of MapReduceClient)
 * @param inputVec a vector of type std::vector<std::pair<K1*, V1*>>, the input elements
 * @param outputVec a vector of type std::vector<std::pair<K3*, V3*>>,
 *                  to which the output elements will be added before returning.
 *                  You can assume that outputVec is empty.
 * @param multiThreadLevel the number of worker threads to be used for running the algorithm
 *                  You can assume multiThreadLevel argument is valid (greater or equal to 1)
 * @return JobHandle that will be used for monitoring the job
 */
JobHandle
startMapReduceJob(const MapReduceClient &client, const InputVec &inputVec, OutputVec &outputVec, int multiThreadLevel) {
    pthread_t threads[multiThreadLevel];
    JobState *job_state = get_new_job_state();
    // NOTICE: I changed it so we build it only once and every thread gets a pointer of it! works the same!
    JobContext *job_context = init_job_context(outputVec, client, job_state,multiThreadLevel);

    ThreadContext *thread_contexts = init_thread_contexts(outputVec, multiThreadLevel, client, inputVec, job_state, job_context);
    for (int i = 0; i < multiThreadLevel; ++i) {
        pthread_create(threads + i, nullptr, map_reduce_method, thread_contexts + i);
    }

    for (int i = 0; i < multiThreadLevel; ++i) {
        pthread_join(threads[i], nullptr);
    }
    int counter = 0;
    for (const auto &pair: inputVec) {
        client.map(pair.first, pair.second, thread_contexts + counter);
        counter++;
    }

    return static_cast<JobHandle>(job_state);
}

void init_thread_context(const InputVec &inputVec, ThreadContext *thread_context,
                         JobContext *job_context, int i);

ThreadContext *init_thread_contexts(OutputVec &outputVec, int multiThreadLevel, const MapReduceClient &client,
                                    const InputVec &inputVec, JobState *job_state, JobContext * job_context) {
    auto *thread_contexts = (ThreadContext *) malloc(multiThreadLevel * sizeof(ThreadContext));

    for (int i = 0; i < multiThreadLevel; ++i) {
        init_thread_context(inputVec, thread_contexts + i, job_context, i);
    }
    return thread_contexts;
}

void init_thread_context(const InputVec &inputVec, ThreadContext *thread_context,
                         JobContext *job_context, const int i) {
    thread_context->thread_id = i;
    thread_context->job_context = job_context;
    thread_context->input_vec = &inputVec;
    thread_context->intermediate_vec = new IntermediateVec();
    job_context->all_intermediate_vec->push_back(thread_context->intermediate_vec);
}

JobContext *init_job_context(OutputVec &outputVec, const MapReduceClient &client, JobState *job_state, int numOfThreads) {
    auto *map_atomic_counter = (std::atomic<int> *) malloc(sizeof(std::atomic<int>));
    map_atomic_counter->store(0);
    auto *reduce_atomic_counter = (std::atomic<int> *) malloc(sizeof(std::atomic<int>));
    reduce_atomic_counter->store(0);
    auto *all_vectors = new std::vector<IntermediateVec *>();
    auto *shuffled_intermediate_vec = new std::vector<IntermediateVec *>();
    auto *job_context = (JobContext *) malloc(sizeof(JobContext));

    job_context->client = &client;
    job_context->all_intermediate_vec = all_vectors;
    job_context->shuffled_intermediate_vec = shuffled_intermediate_vec;
    job_context->output_vec = &outputVec;
    job_context->end_map_stage_mutex = (std::mutex*)malloc(sizeof(std::mutex));
    job_context->job_state = job_state;
    job_context->barrier = new Barrier(numOfThreads);
    job_context->map_atomic_counter = map_atomic_counter;
    job_context->reduce_atomic_counter = reduce_atomic_counter;
    return job_context;
}

JobState *get_new_job_state() {
    auto *job = (JobState *) malloc(sizeof(JobState));
    job->stage = UNDEFINED_STAGE;
    job->percentage = 0;
    return job;
}


/**
 * gets a JobHandle and updates the state of the job into the given JobState struct
 * @param job JobHandle
 * @param state JobState struct to update the job to
 */
void getJobState(JobHandle job, JobState *state) {
    auto job_state = (JobState*)(job);
    state->stage = job_state->stage;
    state->percentage = job_state->percentage;
}

/**
 * Releasing all resources of a job.
 * After this function is called the job handle will be invalid.
 * In case that the function is called and the job is not finished yet wait until the job is finished to close it
 * @param job JobHandle
 */
void closeJobHandle(JobHandle job) {

}

/**
 * produces a (K2*, V2*) pair.
 * updates the number of intermediary elements using atomic counter
 * @param key of input intermediary element
 * @param value of input intermediary element
 * @param context data structure of the thread that created the intermediary element
 */
void emit2(K2 *key, V2 *value, void *context) {
    auto *tc = (ThreadContext *) context;
    IntermediatePair pair = IntermediatePair(key, value);
    tc->intermediate_vec->push_back(pair);
}


/**
 * produces a (K3*, V3*) pair.
 * saves the output element in the context data structures
 * updates the number of output elements using atomic counter
 * @param key of input output element
 * @param value of input output element
 * @param context data structure of the thread that created the output element
 */
void emit3(K3 *key, V3 *value, void *context) {
    auto *tc = (ThreadContext *) context;
    OutputPair pair = OutputPair(key, value);
    tc->job_context->output_vec->push_back(pair);
}

/**
 * gets the JobHandle returned by startMapReduceFramework and waits until it is finished
 * @param job the JobHandle returned by startMapReduceFramework
 */
void waitForJob(JobHandle job);