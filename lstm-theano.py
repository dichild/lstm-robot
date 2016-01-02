import numpy as np
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip
import time
import operator

class RLTheano:
    
    def __init__(self, img_dim, hidden_dim=128, bptt_truncate=-1):
        # Assign instance variables
        self.img_dim = img_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Initialize the network parameters
        E = np.random.uniform(-np.sqrt(1./img_dim), np.sqrt(1./img_dim), (hidden_dim, img_dim))
        U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (8, hidden_dim, hidden_dim)) #6->8
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (8, hidden_dim, hidden_dim)) #6->8
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (7, hidden_dim))
        b = np.zeros((11, hidden_dim)) #6->11
        c = np.zeros(7)
        Do = np.random.uniform(-np.sqrt(1./img_dim), np.sqrt(1./img_dim), (img_dim, hidden_dim))
        Dg = np.random.uniform(-np.sqrt(1./img_dim), np.sqrt(1./img_dim), (img_dim, hidden_dim))
        Din = np.random.uniform(-np.sqrt(1./img_dim), np.sqrt(1./img_dim), (2*hidden_dim, hidden_dim))
        
        # Theano: Created shared variables
        self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))
        self.Do = theano.shared(name='Do', value=Do.astype(theano.config.floatX))
        self.Dg = theano.shared(name='Dg', value=Dg.astype(theano.config.floatX))
        self.Din = theano.shared(name='Din', value=Din.astype(theano.config.floatX))
        # SGD / rmsprop: Initialize parameters
        self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))
        self.mDo = theano.shared(name='mDo', value=np.zeros(Do.shape).astype(theano.config.floatX))
        self.mDg = theano.shared(name='mDg', value=np.zeros(Dg.shape).astype(theano.config.floatX))
        self.mDin = theano.shared(name='mDin', value=np.zeros(Din.shape).astype(theano.config.floatX))
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
    
    def __theano_build__(self):
        E, V, U, W, b, c, Do, Dg, Din= self.E, self.V, self.U, self.W, self.b, self.c, self.Do, self.Dg, self.Din
        
        x = T.ivector('x')
        y = T.ivector('y')
        
        def forward_prop_step(x_t, s_t1_prev, s_t2_prev):
            # This is how we calculated the hidden state in a simple RNN. No longer!
            # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))
            
            # Word embedding layer
#             x_e = E[:,x_t] 
#             # GRU Layer 1
#             z_t1 = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])
#             r_t1 = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])
#             c_t1 = T.tanh(U[2].dot(x_e) + W[2].dot(s_t1_prev * r_t1) + b[2])
#             s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev
            
#             # GRU Layer 2
#             z_t2 = T.nnet.hard_sigmoid(U[3].dot(s_t1) + W[3].dot(s_t2_prev) + b[3])
#             r_t2 = T.nnet.hard_sigmoid(U[4].dot(s_t1) + W[4].dot(s_t2_prev) + b[4])
#             c_t2 = T.tanh(U[5].dot(s_t1) + W[5].dot(s_t2_prev * r_t2) + b[5])
#             s_t2 = (T.ones_like(z_t2) - z_t2) * c_t2 + z_t2 * s_t2_prev
            
            d_o = T.nnet.relu(Do.dot(x_g)+b[8])
            d_g = T.nnet.relu(Dg.dot(x_o)+b[9])
            d_c = T.concatenate([d_o,d_g])
            x_e = T.nnet.relu(Din.dot(d_c)+b[10])
            #     LSTM layer 1
            i_ly1 = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])
            f_ly1 = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])
            o_ly1 = T.nnet.hard_sigmoid(U[2].dot(x_e) + W[2].dot(s_t1_prev) + b[2])
            g_ly1 = T.tanh(U[3].dot(x_e) + W[3].dot(s_t1_prev) + b[3])
            c_ly1 = c_ly1_prev*f_ly1+g_ly1*i_ly1
            s_ly1 = T.tanh(c_ly1)*o_ly1
            
            #     LSTM layer 2        
            i_ly2 = T.nnet.hard_sigmoid(U[4].dot(s_ly1) + W[4].dot(s_t2_prev) + b[4])
            f_ly2 = T.nnet.hard_sigmoid(U[5].dot(s_ly1) + W[5].dot(s_t2_prev) + b[5])
            o_ly2 = T.nnet.hard_sigmoid(U[6].dot(s_ly1) + W[6].dot(s_t2_prev) + b[6])
            g_ly2 = T.tanh(U[7].dot(x_e) + W[7].dot(s_t2_prev) + b[7])
            c_ly2 = c_ly2_prev*f_ly2+g_ly2*i_ly2
            s_ly2 = T.tanh(c_ly2)*o_ly2
            
            
            
            # Final output calculation
            # Theano's softmax returns a matrix with one row, we only need the row
            o_t = T.nnet.softmax(V.dot(s_ly2) + c)[0]

            return [o_t, s_t1, s_t2]
        
        [o, s, s2], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None, 
                          dict(initial=T.zeros(self.hidden_dim)),
                          dict(initial=T.zeros(self.hidden_dim))])
        
        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
        
        # Total cost (could add regularization here)
        cost = o_error
        
        # Gradients
        dE = T.grad(cost, E)
        dU = T.grad(cost, U)
        dW = T.grad(cost, W)
        db = T.grad(cost, b)
        dV = T.grad(cost, V)
        dc = T.grad(cost, c)
        dDo = T.grad(cost,Do) 
        dDg = T.grad(cost,Dg)
        dDin = T.grad(cost,Din)
        
        # Assign functions
        self.predict = theano.function([x], o)
        self.predict_class = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], cost)
        self.bptt = theano.function([x, y], [dE, dU, dW, db, dV, dc, dDo, dDg, dDin])
        
        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')
        
        # rmsprop cache updates
        mE = decay * self.mE + (1 - decay) * dE ** 2
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2
        mb = decay * self.mb + (1 - decay) * db ** 2
        mc = decay * self.mc + (1 - decay) * dc ** 2
        mDo = decay * self.mDo + (1 - decay) * dDo ** 2
        mDg = decay * self.mDg + (1 - decay) * dDg ** 2
        mDin = decay * self.mDin + (1 - decay) * dDin ** 2

        self.sgd_step = theano.function(
            [x, y, learning_rate, theano.Param(decay, default=0.9)],
            [], 
            updates=[(E, E - learning_rate * dE / T.sqrt(mE + 1e-6)),
                     (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                     (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                     (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                     (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
                     (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
                     (Do, Do - learning_rate * dDo / T.sqrt(mDo + 1e-6)),
                     (Dg, Dg - learning_rate * dDg / T.sqrt(mDg + 1e-6)),
                     (Din, Din - learning_rate * dDin / T.sqrt(mDin + 1e-6)),
                     (self.mE, mE),
                     (self.mU, mU),
                     (self.mW, mW),
                     (self.mV, mV),
                     (self.mb, mb),
                     (self.mc, mc),
                     (self.mDo, mDo),
                     (self.mDg, mDg),
                     (self.mDin, mDin)
                    ])
        
        
    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)

