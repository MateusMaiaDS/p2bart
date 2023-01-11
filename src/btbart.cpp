#include "btbart.h"
#include <random>
#include <Rcpp.h>
using namespace std;

// Initialising the model Param
modelParam::modelParam(arma::mat x_train_,
                       arma::vec y_,
                       arma::mat x_test_,
                       int n_tree_,
                       double alpha_,
                       double beta_,
                       double tau_mu_,
                       double tau_,
                       double a_tau_,
                       double d_tau_,
                       double n_mcmc_,
                       double n_burn_){

        // Assign the variables
        x_train = x_train_;
        y = y_;
        x_test = x_test_;
        n_tree = n_tree_;
        alpha = alpha_;
        beta = beta_;
        tau_mu = tau_mu_;
        tau = tau_;
        a_tau = a_tau_;
        d_tau = d_tau_;
        n_mcmc = n_mcmc_;
        n_burn = n_burn_;

}

// Initialising a node
Node::Node(modelParam &data){
        isLeaf = true;
        isRoot = true;
        left = NULL;
        right = NULL;
        parent = NULL;

        train_index = new int[data.x_train.n_rows];
        test_index = new int[data.x_test.n_rows] ;

        // Initialising those vectors
        for(int i = 0; i< data.x_train.n_rows; i ++ ) {
                train_index[i] = -1;
        }
        for(int i = 0; i< data.x_test.n_rows; i ++ ) {
                test_index[i] = -1;
        }


        var_split = 0;
        var_split_rule = 0.0;
        lower = 0.0;
        upper = 1.0;
        curr_weight = 0.0;
        mu = 0.0;
        n_leaf = 0.0;
        r_sq_sum = 0.0;
        r_sum = 0.0;
        log_likelihood = 0.0;
        depth = 0;

}

Node::~Node() {
        if(!isLeaf) {
                delete left;
                delete right;
                delete[] train_index;
                delete[] test_index;
        }
}

// Initializing a stump
void Node::Stump(modelParam& data){

        // Changing the left parent and right nodes;
        left = this;
        right = this;
        parent = this;

        // Updating the training index with the current observations
        for(int i=0; i<data.x_train.n_rows;i++){
                train_index[i] = i;
        }

        // Updating the same for the test observations
        for(int i=0; i<data.x_test.n_rows;i++){
                test_index[i] = i;
        }

}

void Node::addingLeaves(modelParam& data){

     // Create the two new nodes
     left = new Node(data); // Creating a new vector object to the
     right = new Node(data);
     isLeaf = false;

     // Modifying the left node
     left -> isRoot = false;
     left -> isLeaf = true;
     left -> left = left;
     left -> right = left;
     left -> parent = this;
     left -> var_split = 0;
     left -> var_split_rule = 0.0;
     left -> lower = 0.0;
     left -> upper = 1.0;
     left -> mu = 0.0;
     left -> r_sq_sum = 0.0;
     left -> r_sum = 0.0;
     left -> log_likelihood = 0.0;
     left -> n_leaf = 0.0;
     left -> depth = depth+1;

     right -> isRoot = false;
     right -> isLeaf = true;
     right -> left = right; // Recall that you are saving the address of the right node.
     right -> right = right;
     right -> parent = this;
     right -> var_split = 0;
     right -> var_split_rule = 0.0;
     right -> lower = 0.0;
     right -> upper = 1.0;
     right -> mu = 0.0;
     right -> r_sq_sum = 0.0;
     right -> r_sum = 0.0;
     right -> log_likelihood = 0.0;
     right -> n_leaf = 0.0;
     right -> depth = depth+1;

     return;

}

// Creating boolean to check if the vector is left or right
bool Node::isLeft(){
        return (this == this->parent->left);
}

bool Node::isRight(){
        return (this == this->parent->right);
}

// Sample var
void Node::sampleSplitVar(int p){

        // Sampling one index from 0:(p-1)
        var_split = std::rand()%p;
        var_split = 0;

}
// This functions will get and update the current limits for this current variable
void Node::getLimits(){

        // Creating  a new pointer for the current node
        Node* x = this;
        // Already defined this -- no?
        lower = 0.0;
        upper = 1.0;
        // First we gonna check if the current node is a root or not
        bool tree_iter = x->isRoot ? false: true;
        while(tree_iter){
                bool is_left = x->isLeft(); // This gonna check if the current node is left or not
                x = x->parent; // Always getting the parent of the parent
                tree_iter = x->isRoot ? false : true; // To stop the while
                if(x->var_split == var_split){
                        tree_iter = false ; // This stop is necessary otherwise we would go up til the root, since we are always update there is no prob.
                        if(is_left){
                                upper = x->var_split_rule;
                                lower = x->lower;
                        } else {
                                upper = x->upper;
                                lower = x->var_split_rule;
                        }
                }
        }
}


void Node::displayCurrNode(){

                std::cout << "Node address: " << this << std::endl;
                std::cout << "Node parent: " << parent << std::endl;

                std::cout << "Cur Node is leaf: " << isLeaf << std::endl;
                std::cout << "Cur Node is root: " << isRoot << std::endl;
                std::cout << "Cur The split_var is: " << var_split << std::endl;
                std::cout << "Cur The split_var_rule is: " << var_split_rule << std::endl;

                return;
}


void Node::deletingLeaves(){

     // Should I create some warn to avoid memoery leak
     //something like it will only delete from a nog?
     // Deleting
     delete left; // This release the memory from the left point
     delete right; // This release the memory from the right point
     left = this;  // The new pointer for the left become the node itself
     right = this; // The new pointer for the right become the node itself
     isLeaf = true;

     return;

}
// Getting the leaves (this is the function that gonna do the recursion the
//                      function below is the one that gonna initialise it)
void get_leaves(Node* x,  std::vector<Node*> &leaves_vec) {

        if(x->isLeaf){
                leaves_vec.push_back(x);
        } else {
                get_leaves(x->left, leaves_vec);
                get_leaves(x->right,leaves_vec);
        }

        return;

}



// Initialising a vector of nodes in a standard way
std::vector<Node*> leaves(Node* x) {
        std::vector<Node*> leaves_init(0); // Initialising a vector of a vector of pointers of nodes of size zero
        get_leaves(x,leaves_init);
        return(leaves_init);
}

// Sweeping the trees looking for nogs
void get_nogs(std::vector<Node*>& nogs, Node* node){
        if(!node->isLeaf){
                bool bool_left_is_leaf = node->left->isLeaf;
                bool bool_right_is_leaf = node->right->isLeaf;

                // Checking if the current one is a NOGs
                if(bool_left_is_leaf && bool_right_is_leaf){
                        nogs.push_back(node);
                } else { // Keep looking for other NOGs
                        get_nogs(nogs, node->left);
                        get_nogs(nogs, node->right);
                }
        }
}

// Creating the vectors of nogs
std::vector<Node*> nogs(Node* tree){
        std::vector<Node*> nogs_init(0);
        get_nogs(nogs_init,tree);
        return nogs_init;
}



// Initializing the forest
Forest::Forest(modelParam& data){

        // Creatina vector of size of number of trees
        trees.resize(data.n_tree);
        for(int  i=0;i<data.n_tree;i++){
                // Creating the stump for each tree
                trees[i] = new Node(data);
                // Filling up each stump for each tree
                trees[i]->Stump(data);
        }
}

// Function to delete one tree
Forest::~Forest(){
        for(int  i=0;i<trees.size();i++){
                delete trees[i];
        }
}

// Selecting a random node
Node* sample_node(std::vector<Node*> leaves_){

        // Getting the number of leaves
        int n_leaves = leaves_.size();
        return(leaves_[std::rand()%n_leaves]);
}

// Grow a tree for a given rule
void grow(Node* tree, modelParam &data, arma::vec &curr_res){

        // Getting the number of terminal nodes
        std::vector<Node*> t_nodes = leaves(tree) ;
        std::vector<Node*> nog_nodes = nogs(tree);

        // Selecting one node to be sampled
        Node* g_node = sample_node(t_nodes);

        // Store all old quantities that will be used or not
        double old_lower = g_node->lower;
        double old_upper = g_node->upper;
        int old_var_split = g_node->var_split;
        double old_var_split_rule = g_node->var_split_rule;

        // Calculate current tree log likelihood
        double tree_log_like = 0;

        // Calculating the whole likelihood fo the tree
        for(int i = 0; i < t_nodes.size(); i++){
                t_nodes[i]->nodeLogLike(data, curr_res);
                tree_log_like = tree_log_like + t_nodes[i]->log_likelihood;
        }

        // Adding the leaves
        g_node->addingLeaves(data);

        // Selecting the var
        g_node-> sampleSplitVar(data.x_train.n_cols);
        // Updating the limits
        g_node->getLimits();
        // Selecting a rule
        g_node->var_split_rule = (g_node->upper-g_node->lower)*rand_unif()+g_node->lower;


        // Create an aux for the left and right index
        int train_left_counter = 0;
        int train_right_counter = 0;

        int test_left_counter = 0;
        int test_right_counter = 0;


        // Updating the left and the right nodes
        for(int i = 0;i<data.x_train.n_rows;i++){
                if(g_node -> train_index[i] == -1){
                        break;
                }
                if(data.x_train(g_node->train_index[i],g_node->var_split)<g_node->var_split_rule){
                        g_node->left->train_index[train_left_counter] = g_node->train_index[i];
                        train_left_counter++;
                } else {
                        g_node->right->train_index[train_right_counter] = g_node->train_index[i];
                        train_right_counter++;
                }
        }

        // Updating the left and right nodes for the
        for(int i = 0;i<data.x_test.n_rows; i++){
                if(g_node -> test_index[i] == -1){
                        break;
                }
                if(data.x_test(g_node->test_index[i],g_node->var_split)<g_node->var_split_rule){
                        g_node->left->test_index[test_left_counter] = g_node->test_index[i];
                        test_left_counter++;
                } else {
                        g_node->right->test_index[test_right_counter] = g_node->test_index[i];
                        test_right_counter++;
                }
        }

        // Updating the loglikelihood for those terminal nodes
        g_node->left->nodeLogLike(data, curr_res);
        g_node->right->nodeLogLike(data, curr_res);


        // Calculating the prior term for the grow
        double tree_prior = log(data.alpha*pow((1+g_node->depth),-data.beta)) +
                log(1-data.alpha*pow((1+g_node->depth+1),-data.beta)) + // Prior of left node being terminal
                log(1-data.alpha*pow((1+g_node->depth+1),-data.beta)) - // Prior of the right noide being terminal
                log(1-data.alpha*pow((1+g_node->depth),-data.beta)); // Old current node being terminal

        // Getting the transition probability
        double log_transition_prob = log((0.3)/(nog_nodes.size()+1)) - log(0.3/t_nodes.size()); // 0.3 and 0.3 are the prob of Prune and Grow, respectively

        // Calculating the loglikelihood for the new branches
        double new_tree_log_like = tree_log_like - g_node->log_likelihood + g_node->left->log_likelihood + g_node->right->log_likelihood ;

        // Calculating the acceptance ratio
        double acceptance = exp(new_tree_log_like - tree_log_like + log_transition_prob + tree_prior);

        // Keeping the new tree or not
        if(rand_unif()<acceptance){
                // Do nothing just keep the new tree
        } else {
                // Returning to the old values
                g_node->var_split = old_var_split;
                g_node->var_split_rule = old_var_split_rule;
                g_node->lower = old_lower;
                g_node->upper = old_upper;
                g_node->deletingLeaves();
        }

        return;

}


// Pruning a tree
void prune(Node* tree, modelParam&data, arma::vec &curr_res){


        // Getting the number of terminal nodes
        std::vector<Node*> t_nodes = leaves(tree);

        // Can't prune a root
        if(t_nodes.size()<2){
                return;
        }

        std::vector<Node*> nog_nodes = nogs(tree);

        // Selecting one node to be sampled
        Node* p_node = sample_node(nog_nodes);


        // Calculate current tree log likelihood
        double tree_log_like = 0;

        // Calculating the whole likelihood fo the tree
        for(int i = 0; i < t_nodes.size(); i++){
                t_nodes[i]->nodeLogLike(data, curr_res);
                tree_log_like = tree_log_like + t_nodes[i]->log_likelihood;
        }

        // Updating the loglikelihood of the selected pruned node
        p_node->nodeLogLike(data, curr_res);

        // Getting the loglikelihood of the new tree
        double new_tree_log_like = tree_log_like + p_node->log_likelihood - (p_node->left->log_likelihood + p_node->right->log_likelihood);

        // Calculating the transition loglikelihood
        double transition_loglike = log((0.3)/(t_nodes.size())) - log((0.3)/(nog_nodes.size()));

        // Calculating the prior term for the grow
        double tree_prior = log(1-data.alpha*pow((1+p_node->depth),-data.beta))-
                log(data.alpha*pow((1+p_node->depth),-data.beta)) -
                log(1-data.alpha*pow((1+p_node->depth+1),-data.beta)) - // Prior of left node being terminal
                log(1-data.alpha*pow((1+p_node->depth+1),-data.beta));  // Prior of the right noide being terminal
                 // Old current node being terminal


        // Calculating the acceptance
        double acceptance = exp(new_tree_log_like - tree_log_like + transition_loglike + tree_prior);

        if(rand_unif()<acceptance){
                p_node->deletingLeaves();
        }

        return;
}


// Creating the change verb
void change(Node* tree, modelParam &data, arma::vec &curr_res){


        // Setting the size of the tree
        if(tree->isRoot) {
                return;
        }

        // Getting the number of terminal nodes
        std::vector<Node*> t_nodes = leaves(tree) ;
        std::vector<Node*> nog_nodes = nogs(tree);

        // Selecting one node to be sampled
        Node* c_node = sample_node(nog_nodes);

        // Calculate current tree log likelihood
        double tree_log_like = 0;

        // Calculating the whole likelihood fo the tree
        for(int i = 0; i < t_nodes.size(); i++){
                // cout << "Loglike error " << ed
                t_nodes[i]->nodeLogLike(data, curr_res);
                tree_log_like = tree_log_like + t_nodes[i]->log_likelihood;
        }

        // If the current node has size zero there is no point of change its rule
        if(c_node->n_leaf==0) {
                return;
        }

        // Storing all the old loglikelihood from left
        double old_left_log_like = c_node->left->log_likelihood;
        double old_left_r_sum = c_node->left->r_sum;
        double old_left_r_sq_sum = c_node->left->r_sq_sum;
        double old_left_n_leaf = c_node->left->n_leaf;
        int old_left_train_index[data.x_train.n_rows];

        for(int i = 0; i < data.x_train.n_rows;i++){
                old_left_train_index[i] = c_node->left->train_index[i];
                c_node->left->train_index[i] = -1;
        }

        // Storing all of the old loglikelihood from right;
        double old_right_log_like = c_node->right->log_likelihood;
        double old_right_r_sum = c_node->right->r_sum;
        double old_right_r_sq_sum = c_node->right->r_sq_sum;
        double old_right_n_leaf = c_node->right->n_leaf;
        int old_right_train_index[data.x_train.n_rows];

        for(int i = 0; i< data.x_train.n_rows;i++){
                old_right_train_index[i] = c_node->right->test_index[i];
                c_node->right->train_index[i] = -1;
        }


        // Storing test observations
        int old_left_test_index[data.x_test.n_rows];
        int old_right_test_index[data.x_test.n_rows];

        for(int i = 0; i< data.x_test.n_rows;i++){
                old_left_test_index[i] = c_node->left->test_index[i];
                c_node->left->test_index[i] = -1;
        }

        for(int i = 0; i< data.x_test.n_rows;i++){
                old_right_test_index[i] = c_node->right->test_index[i];
                c_node->right->test_index[i] = -1;
        }

        // Storing the old ones
        int old_var_split = c_node->var_split;
        int old_var_split_rule = c_node->var_split_rule;
        int old_lower = c_node->lower;
        int old_upper = c_node->upper;

        // Selecting the var
        c_node-> sampleSplitVar(data.x_train.n_cols);
        // Updating the limits
        c_node->getLimits();
        // Selecting a rule
        c_node -> var_split_rule = (c_node->upper-c_node->lower)*rand_unif()+c_node->lower;

        // Create an aux for the left and right index
        int train_left_counter = 0;
        int train_right_counter = 0;

        int test_left_counter = 0;
        int test_right_counter = 0;


        // Updating the left and the right nodes
        for(int i = 0;i<data.x_train.n_rows;i++){
                // cout << " Train indexeses " << c_node -> train_index[i] << endl ;
                if(c_node -> train_index[i] == -1){
                        break;
                }
                // cout << " Current train index " << c_node->train_index[i] << endl;

                if(data.x_train(c_node->train_index[i],c_node->var_split)<c_node->var_split_rule){
                        c_node->left->train_index[train_left_counter] = c_node->train_index[i];
                        train_left_counter++;
                } else {
                        c_node->right->train_index[train_right_counter] = c_node->train_index[i];
                        train_right_counter++;
                }
        }

        // Updating the left and the right nodes
        for(int i = 0;i<data.x_train.n_rows;i++){
                if(c_node -> test_index[i] == -1){
                        break;
                }
                if(data.x_test(c_node->test_index[i],c_node->var_split)<c_node->var_split_rule){
                        c_node->left->test_index[test_left_counter] = c_node->test_index[i];
                        test_left_counter++;
                } else {
                        c_node->right->test_index[test_right_counter] = c_node->test_index[i];
                        test_right_counter++;
                }
        }

        // Updating the new left and right loglikelihoods
        c_node->left->nodeLogLike(data,curr_res);
        c_node->right->nodeLogLike(data,curr_res);


        // Calculating the acceptance
        double new_tree_log_like =  - old_left_log_like - old_right_log_like + c_node->left->log_likelihood + c_node->right->log_likelihood;

        double acceptance = exp(new_tree_log_like);

        if(rand_unif()<acceptance){
                // Keep all the treesi
        } else {

                // Returning to the previous values
                c_node->var_split = old_var_split;
                c_node->var_split_rule = old_var_split_rule;
                c_node->lower = old_lower;
                c_node->upper = old_upper;

                // Returning to the old ones
                c_node->left->r_sum = old_left_r_sum;
                c_node->left->r_sq_sum = old_left_r_sq_sum;
                c_node->left->n_leaf = old_left_n_leaf;
                c_node->left->log_likelihood = old_left_log_like;
                c_node->left->train_index = old_left_train_index;
                c_node->left->test_index = old_left_test_index;

                c_node->right->r_sum = old_right_r_sum;
                c_node->right->r_sq_sum = old_right_r_sq_sum;
                c_node->right->n_leaf = old_right_n_leaf;
                c_node->right->log_likelihood = old_right_log_like;
                c_node->right->train_index = old_right_train_index;
                c_node->right->test_index = old_right_test_index;

        }

        return;
}

// Calculating the Loglilelihood of a node
void Node::nodeLogLike(modelParam& data, arma::vec &curr_res){


        // When we generate empty nodes we don't want to accept them;
        if(train_index[0]==-1){
                r_sum = 0;
                r_sq_sum = 10000;
                n_leaf = 0;
                log_likelihood = -2000000; // Absurd value avoid this case
                return;
        }

        r_sum = 0;
        r_sq_sum = 0;
        n_leaf = 0;

        for(int i = 0; i<data.x_train.n_rows; i++){

                // Exiting before
                if(train_index[i]==-1){
                        break;
                }

                // Calculating node quantities
                r_sum = r_sum + curr_res(train_index[i]);
                r_sq_sum = r_sq_sum + curr_res(train_index[i])*curr_res(train_index[i]);
                n_leaf = n_leaf + 1;

        }

        log_likelihood = - 0.5*data.tau*r_sq_sum - 0.5*log(data.tau_mu + (n_leaf*data.tau)) + (0.5*(data.tau*data.tau)*(r_sum*r_sum))/( (data.tau*n_leaf)+data.tau_mu);

        return;

}


// Update mu
void updateMu(Node* tree, modelParam &data){

        // Getting the number of terminal nodes
        std::vector<Node*> t_nodes = leaves(tree) ;

        // Iterating over the terminal nodes and updating it its prediction
        for(int i = 0;i < t_nodes.size();i++){
                t_nodes[i]->mu = R::rnorm((data.tau*t_nodes[i]->r_sum)/(t_nodes[i]->n_leaf*data.tau+data.tau_mu),sqrt(1/(data.tau*t_nodes[i]->n_leaf+data.tau_mu))) ;
        }
}

// Get the prediction
void getPredictions(Node* tree,
                    modelParam data,
                    arma::vec& current_prediction_train,
                    arma::vec current_prediction_test){

        // Getting the current prediction
        vector<Node*> t_nodes = leaves(tree);
        for(int i = 0; i<t_nodes.size();i++){

                // For the training samples
                for(int j = 0; j<data.x_train.n_rows; j++){
                        if((t_nodes[i]->train_index[j])==-1.0){
                                break;
                        }
                        current_prediction_train[t_nodes[i]->train_index[j]] = t_nodes[i]->mu;
                }

                // Regarding the test samples
                for(int j = 0; j< data.x_test.n_rows;j++){

                        if(t_nodes[i]->test_index[j]==-1){
                                break;
                        }
                }
        }
}

// Updating the tau parameter
void updateTau(arma::vec &y_hat,
               modelParam &data){

        // Getting the sum of residuals square
        double tau_res_sq_sum = dot((y_hat-data.y),(y_hat-data.y));

        data.tau = R::rgamma((0.5*data.y.size()+data.a_tau),1/(0.5*tau_res_sq_sum+data.d_tau));

        return;
}

// [[Rcpp::export]]
// Creating the BART function
Rcpp::List bart(arma::mat x_train,
          arma::vec y_train,
          arma::mat x_test,
          int n_tree,
          int n_mcmc,
          int n_burn,
          double tau, double mu,
          double tau_mu,
          double alpha, double beta,
          double a_tau, double d_tau){

        // Posterior counter
        int curr = 0;
        // Creating the structu object
        modelParam data(x_train,
                        y_train,
                        x_test,
                        n_tree,
                        alpha,
                        beta,
                        tau_mu,
                        tau,
                        a_tau,
                        d_tau,
                        n_mcmc,
                        n_burn);

        // Getting the n_post
        int n_post = n_mcmc - n_burn;

        // Defining those elements
        arma::mat y_train_hat_post = arma::zeros<arma::mat>(data.x_train.n_rows,n_post);
        arma::mat y_test_hat_post = arma::zeros<arma::mat>(data.x_train.n_rows,n_post);
        arma::vec tau_post = arma::zeros<arma::vec>(n_post);

        // Defining other variables
        arma::vec partial_pred = arma::zeros<arma::vec>(data.x_train.n_rows);
        arma::vec partial_residuals = arma::zeros<arma::vec>(data.x_train.n_rows);
        arma::mat tree_fits_store = arma::zeros<arma::mat>(data.x_train.n_rows,data.n_tree);
        arma::mat tree_fits_store_test = arma::zeros<arma::mat>(data.x_test.n_rows,data.n_tree);

        // Getting zeros
        arma::vec prediction_train_sum = arma::zeros<arma::vec>(data.x_train.n_rows);
        arma::vec prediction_test_sum = arma::zeros<arma::vec>(data.x_test.n_rows);
        arma::vec prediction_train = arma::zeros<arma::vec>(data.x_train.n_rows);
        arma::vec prediction_test = arma::zeros<arma::vec>(data.x_test.n_rows);

        double verb;

        // Defining progress bars parameters
        const int width = 70;
        double pb = 0;



        // Selecting the train
        Forest all_forest(data);

        for(int i = 0;i<data.n_mcmc;i++){

                // Initialising PB
                std::cout << "[";
                int k = 0;
                // Evaluating progress bar
                for(;k<=pb*width/data.n_mcmc;k++){
                        std::cout << "=";
                }

                for(; k < width;k++){
                        std:: cout << " ";
                }

                std::cout << "] " << std::setprecision(5) << (pb/data.n_mcmc)*100 << "%\r";
                std::cout.flush();


                prediction_train_sum = arma::zeros<arma::vec>(data.x_train.n_rows);
                prediction_test_sum = arma::zeros<arma::vec>(data.x_test.n_rows);

                for(int t = 0; t<data.n_tree;t++){

                        // Updating the partial residuals
                        partial_residuals = data.y-partial_pred+tree_fits_store.col(t);

                        // Iterating over all trees
                        verb = rand_unif();

                        // Selecting the verb
                        if(verb < 0.3){
                                grow(all_forest.trees[t],data,partial_residuals);
                        } else if(verb>=0.3 & verb <0.6) {
                                prune(all_forest.trees[t], data, partial_residuals);
                        } else {
                                change(all_forest.trees[t], data, partial_residuals);

                        }

                        // Updating the Mu
                        updateMu(all_forest.trees[t], data);

                        // Updating the current prediction
                        getPredictions(all_forest.trees[t],data,prediction_train,prediction_test);

                        // Updating the partial pred
                        partial_pred = partial_pred - tree_fits_store.col(t) + prediction_train;

                        tree_fits_store.col(t) = prediction_train;

                        prediction_train_sum = prediction_train_sum + prediction_train;

                        prediction_test_sum = prediction_test_sum + prediction_test;

                        // Updating the Tau
                        updateTau(partial_pred, data);

                }

                if(i > n_burn){
                        // Storing the predictions
                        y_train_hat_post.col(curr) = prediction_train_sum;
                        y_test_hat_post.col(curr) = prediction_test_sum;
                        tau_post(curr) = data.tau;
                        curr++;
                }

                pb += 1;

        }
        // Initialising PB
        std::cout << "[";
        int k = 0;
        // Evaluating progress bar
        for(;k<=pb*width/data.n_mcmc;k++){
                std::cout << "=";
        }

        for(; k < width;k++){
                std:: cout << " ";
        }

        std::cout << "] " << std::setprecision(5) << 100 << "%\r";
        std::cout.flush();

        std::cout << std::endl;

        return Rcpp::List::create(y_train_hat_post,
                                  y_test_hat_post,
                                  tau_post);
}


// TESTING FUNCTIONS

// // Creating a function to create a tree and select terminal nodes to be grown
// // [[Rcpp::export]]
// void createTree(const arma::mat X,
//                 const arma::vec y,
//                 int n_tree){
//
//         modelParam data(X,
//                         y,
//                         X,
//                         10,
//                         0.95,
//                         2.0,
//                         1.0,
//                         1.0,
//                         1.0,
//                         1.0,
//                         2000,
//                         500);
//         // Creating the data object
//         Forest all_trees = Forest(data);
//
//         // Getting the leaves of my first tree
//         for(int k=0;k<10;k++){
//                 std::vector<Node*> tree_zero_leafs = leaves(all_trees.trees[0]);
//                 Node* g_leaf = sample_node(tree_zero_leafs);
//
//                 cout << "Verifying the first tree: " << all_trees.trees[0] <<endl;
//                 cout << "Verifying the first tree: " << g_leaf <<endl;
//
//                 // Displaying this node before grow
//                 g_leaf->displayCurrNode();
//                 // Now I gonna grow the chosen node
//                 g_leaf->grow(tree_zero_leafs[0],data, y);
//                 g_leaf->displayCurrNode();
//
//                 cout << " ================= " << endl;
//                 cout << " ======= " << k <<" ======" << endl;
//                 cout << " ================= " << endl;
//
//         }
//
// }

// Testing likelihood calculation

// [[Rcpp::export]]
double test_logtree(arma::mat X,
                    arma::vec y){

        modelParam data(X,
                        y,
                        X,
                        10,
                        0.95,
                        2.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        2000,
                        500);

        // Creating a tree
        Node node_init(data) ;
        node_init.Stump(data);
        // Forest forest(data);
        node_init.nodeLogLike(data,data.y);
        std::cout << "Root loglike " << node_init.log_likelihood << std::endl;

        // Trying to grow a tree
        grow(&node_init, data, y);
        std::cout << " GROWN TREE - Root loglike: " << node_init.left->log_likelihood << std::endl;
        cout << "Number of leaves: " << leaves(&node_init).size() << endl;
        prune(&node_init, data,  y);
        cout << "Number of leaves: " << leaves(&node_init).size() << endl;
        // Trying to grow a tree
        for(int  i=0;i<100;i++){
                grow(&node_init, data, y);
        }
        std::cout << " GROWN TREE - Root loglike: " << node_init.left->left->log_likelihood << std::endl;
        cout << "Number of leaves: " << leaves(&node_init).size() << endl;

        // Trying to grow a tree
        for(int  i=0;i<100;i++){
                prune(&node_init, data, y);
        }

        // Trying to change a tree
        for(int  j=0;j<100;j++){
                change(&node_init, data, y);
        }
        std::cout << " CHANGE TREE - Root loglike: " << node_init.left->left->log_likelihood << std::endl;
        cout << "Number of leaves: " << leaves(&node_init).size() << endl;

        return 0.0;
}

// // [[Rcpp::export]]
// void adding_two_vec(arma::vec x, arma::vec y) {
//
//         arma::vec z = x+y;
//         // Adding those two values
//         for(int i = 0; i<x.size(); i++) {
//                 std::cout << "Z position" << i << "is given by " << z[i] << std::endl;
//         }
//
//         return;
// }
//
//
// // [[Rcpp::export]]
// void testingDisplay(arma::mat X,
//                     arma::vec y){
//
//         modelParam data(X,
//                         y,
//                         X,
//                         10,
//                         0.95,
//                         2.0,
//                         1.0,
//                         1.0,
//                         1.0,
//                         1.0,
//                         2000,
//                         500);
//
//         Node node_init(data);
//         node_init.displayNode();
//         // node_init.addingLeaves();
//         std::vector<Node*> return_leaves_init  = leaves(&node_init);
//         std::cout << "Final try first " << return_leaves_init.size()<< std::endl;
//
//         // node_init.left->addingLeaves();
//         return_leaves_init  = leaves(&node_init);
//         std::cout << "Final try second " << return_leaves_init.size()<< std::endl;
//
//         node_init.displayNode();
//
//         Node* point_tree = &node_init;
//         // Getting all leaves
//         std::vector<Node*> return_leaves = leaves(point_tree);
//
//         for(int i=0;i<return_leaves.size();i++){
//                 std::cout << "Terminal leaf left: " << return_leaves[i]->left<<std::endl;
//                 std::cout << "Terminal leaf right: " << return_leaves[i]->left<<std::endl;
//
//         }
//         return;
// }
//

// [[Rcpp::export]]
void modifyVector(arma::vec& x) {

        // Modify each element of the vector by sampling a normal value
        for (int i = 0; i < x.size(); i++) {
                x[i] = arma::randn();
        }
}

// [[Rcpp::export]]
arma::vec rTranslate() {

        arma::vec x = arma::zeros<arma::vec>(10);
        modifyVector(x);
        return x;
}


// Testing grow a tree
// [[Rcpp::export]]
int unit_test_grow(arma::mat x_train,
                   arma::vec y_train){


        // Growing infinitely the tree
        modelParam data(x_train,
                        y_train,
                        x_train,
                        200,
                        0.95,
                        2,
                        1,
                        1,
                        1,
                        1,
                        2000,
                        100);


        // Creating a Forest
        Forest all_trees(data);

        // Iterating over multiple MCMC tests
        for(int i=0; i < 1000; i++){
                        for(int t = 0; t < data.n_tree; t++){
                                grow(all_trees.trees[t],data,data.y);
                                prune(all_trees.trees[t],data,data.y);
                                change(all_trees.trees[t],data,data.y);

                        }
        }


        return 0;
}
