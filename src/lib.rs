use std::io::{Write, Error, BufRead, BufReader, BufWriter};
use std::fs::*;
use pyo3::prelude::*;
use rand::prelude::*;

type Float = f32;

trait VectorData {
    fn get_data(&self) -> &[Float];
    fn get_data_mut(&mut self) -> &mut[Float];
}

trait Vector: VectorData {
    fn copy(&self) -> PyVector;
    fn add_assign<V: VectorData>(&mut self, other: &V);
    fn mul_assign(&mut self, lambda: Float);
    fn add_mul_assign<V: VectorData>(&mut self, other: &V, lambda: Float);
    fn dot<V: VectorData>(&self, other: &V) -> Float;
}

pub struct RefVector<'a> {
    data: &'a mut[Float]
}

impl<'a> VectorData for RefVector<'a>{
    fn get_data(&self) -> &[Float] {self.data}
    fn get_data_mut(&mut self) -> &mut[Float] {self.data}
}

#[derive(Clone)]
#[pyclass]
#[repr(align(64))]
pub struct PyVector {
    #[pyo3(get, set)]
    data: Vec<Float>
}

impl VectorData for PyVector {
    fn get_data(&self) -> &[Float] {
        &self.data[..]
    }
    fn get_data_mut(&mut self) -> &mut[Float] {
        &mut self.data[..]
    }
}

impl<V: VectorData> Vector for V {
    fn copy(&self) -> PyVector { 
        PyVector { data: self.get_data().to_vec()}
    }

    fn add_assign<Vo: VectorData>(&mut self, other: &Vo) {
        debug_assert_eq!(self.get_data().len(), other.get_data().len());
        for i  in 0..self.get_data().len(){
            self.get_data_mut()[i] += other.get_data()[i];
        }
    }

    fn mul_assign(&mut self, lambda: Float) {
        for i in 0..self.get_data().len() {
            self.get_data_mut()[i] *= lambda;
        }
    }

    fn add_mul_assign<Vo: VectorData>(&mut self, other: &Vo, lambda: Float) {
        debug_assert_eq!(self.get_data().len(), other.get_data().len());
        for i  in 0..self.get_data().len(){
            self.get_data_mut()[i] += other.get_data()[i] * lambda;
        } 
    }

    fn dot<Vo: VectorData>(&self, other: &Vo) -> Float {
        debug_assert_eq!(self.get_data().len(), other.get_data().len());
        let mut sum: Float = 0.;
        for i  in 0..self.get_data().len(){
            sum += self.get_data()[i] * other.get_data()[i];
        }
        sum
    }
}

#[pymethods]
impl PyVector {
    #[new]
    pub fn new(size: usize) -> Self {
        let mut vec = PyVector {
            data: Vec::with_capacity(size)
        };
        vec.data.resize(size, 0.0);
        vec
    }
    
    fn __repr__(&self) -> String {
        let mut out = String::with_capacity(1024);
        if self.data.len() < 10 {
            for n in self.data.iter() {
                let mut num: String = n.to_string();   
                while num.len() < 9 {
                    num += " ";
                }
                out += &(num + "\t");
            }
            out += "\n";
        } else {
            for n in self.data[0..5].iter() {
                let mut num: String = n.to_string();   
                while num.len() < 9 {
                    num += " ";
                }
                out += &(num + "\t");
            }
            out += "... ";
            for n in self.data[(self.data.len()-5)..].iter() {
                let mut num: String = n.to_string();
                while num.len() < 9 {
                    num += " ";
                }
                out += &(num + "\t");
            }
            out += "\n";
        }
        out
    }

    fn __iadd__(&mut self, other: Self) {
        self.add_assign(&other);
    }

    fn __imul__(&mut self, lambda: Float) {
        self.mul_assign(lambda);
    }

    fn __mul__(&self, other: Self) -> Float {
        self.dot(&other)
    }
}

trait MatrixData {
    fn get_data(&self) -> &[Float];
    fn get_data_mut(&mut self) -> &mut[Float];
    fn get_width(&self) -> usize;
}

struct RefMatrix<'a> {
    data: &'a mut[Float],
    width: usize
}

impl<'a> MatrixData for RefMatrix<'a>{
    fn get_data(&self) -> &[Float] {self.data}
    fn get_data_mut(&mut self) -> &mut[Float] {self.data}
    fn get_width(&self) -> usize {self.width}
}

#[pyclass]
#[derive(Clone)]
#[repr(align(64))]
pub struct PyMatrix {
    #[pyo3(get, set)]
    data: Vec<Float>,
    #[pyo3(get)]
    width: usize
}

impl MatrixData for PyMatrix{
    fn get_data(&self) -> &[Float] {&self.data[..]}
    fn get_data_mut(&mut self) -> &mut[Float] {&mut self.data[..]}
    fn get_width(&self) -> usize {self.width}
}

trait Matrix: MatrixData {
    fn clone(&self) -> PyMatrix;
    fn add_assign<M: MatrixData>(&mut self, other: &M);
    fn mul_assign(&mut self, lambda: Float);
    fn dot<Vi: Vector, Vo: Vector>(&mut self, vec: &Vi, out: &mut Vo);
    fn transpose_dot<Vi: Vector, Vo: Vector>(&mut self, vec: &Vi, out: &mut Vo);
}

impl<M: MatrixData> Matrix for M {
    fn clone(&self) -> PyMatrix {
        PyMatrix { data: self.get_data().to_vec(), width: self.get_width() }
    }

    fn add_assign<Mo: MatrixData>(&mut self, other: &Mo) { 
        debug_assert_eq!(self.get_data().len(), other.get_data().len());
        for i  in 0..self.get_data().len(){
            self.get_data_mut()[i] += other.get_data()[i];
        }
    }
    
    fn mul_assign(&mut self, lambda: Float) {
        for i in 0..self.get_data().len() {
            self.get_data_mut()[i] *= lambda;
        }
    }

    fn dot<Vi: Vector, Vo: Vector>(&mut self, vec: &Vi, out: &mut Vo) {
        debug_assert_eq!(self.get_width(), vec.get_data().len());
        debug_assert_eq!(out.get_data().len()*self.get_width(), self.get_data().len());
        for i in 0..out.get_data().len() {
            let a = i * self.get_width();
            let b = a + self.get_width();
            out.get_data_mut()[i] = RefVector{
                data: &mut self.get_data_mut()[a..b]
            }.dot(vec);
        }
    }

    fn transpose_dot<Vi: Vector, Vo: Vector>(&mut self, vec: &Vi, out: &mut Vo) {
        debug_assert_eq!(self.get_width(), out.get_data().len());
        debug_assert_eq!(vec.get_data().len()*self.get_width(), self.get_data().len());
        out.get_data_mut().fill(0.0);
        for i in 0..vec.get_data().len() {
            let a = i * self.get_width();
            let b = a + self.get_width();
            out.add_mul_assign(&mut RefVector {
                data: &mut self.get_data_mut()[a..b]
            }, vec.get_data()[i]);
        }
    }
}

#[pymethods]
impl PyMatrix {
    #[new]
    fn new(width: usize, height: usize) -> Self {
        let mut mat = PyMatrix {
            data: Vec::with_capacity(width * height),
            width
        };
        mat.data.resize(width * height, 0.0);
        mat
    }

    fn __repr__(&self) -> String {
        let mut out = String::with_capacity(2048);
        for i in 0..(self.get_data().len()/self.get_width()) {
            let a = i * self.get_width();
            let b = a + self.get_width();
            out += PyVector{
                data: self.get_data()[a..b].to_vec()
            }.__repr__().as_str();
        }
        out
    }

    fn __iadd__(&mut self, other: Self) {
        self.add_assign(&other);
    }

    fn __imul__(&mut self, lambda: Float) {
        self.mul_assign(lambda);
    }

    fn __mul__(&mut self, vec: PyVector) -> PyVector {
        let mut out = PyVector::new(self.get_data().len()/self.get_width());
        self.dot(&vec, &mut out);
        out
    }

    fn t_dot(&mut self, vec: PyVector) -> PyVector {
        let mut out = PyVector::new(self.get_width());
        self.transpose_dot(&vec, &mut out);
        out
    }
}

trait LearningUnit where Self: Sized {
    fn load<Reader: BufRead>(reader: &mut Reader) -> Result<Self, Error>;
    fn save<Writer: Write>(&self, writer: &mut Writer) -> Result<(), Error>;
    fn get_input_size(&self) -> usize;
    fn get_output_size(&self) -> usize;
    fn forward<Vi: Vector, Vo: Vector>(&mut self, input: &Vi, output: &mut Vo);
    fn calc_error<Vd: Vector, Ve: Vector>(&mut self, diff: &Vd, error: &mut Ve);
    fn backward_diff<Ve: Vector, Vd: Vector>(&mut self, error: &Ve, diff: &mut Vd);
    fn update_weights<Vi: Vector, Ve: Vector>(&mut self, input: &Vi, error: &Ve, rate: Float);
    fn randomize(&mut self, range: Float);
}

trait ActivationFunction {
    fn new() -> Self;
    fn function(x: Float) -> Float;
    fn derive(x: Float) -> Float;
}

struct Sigmoid;
impl ActivationFunction for Sigmoid {
    fn new() -> Self {
        Sigmoid {}
    }

    fn function(x: Float) -> Float {
        1./((-x).exp() + 1.)
    }

    fn derive(x: Float) -> Float {
        let f = Sigmoid::function(x);
        f * (1. - f)
    }
}

struct ReLU;
impl ActivationFunction for ReLU {
    fn new() -> Self {
        ReLU {}
    }

    fn function(x: Float) -> Float {
        if x < 0. {
            0.
        } else {
            x
        }
    }

    fn derive(x: Float) -> Float {
        if x < 0. {
            0.
        } else {
            1.
        }
    }
}

struct Perceptron<G: ActivationFunction> {
    weights: PyMatrix,
    bias: PyVector,
    input_size: usize,
    output_size: usize,
    h: PyVector,
    _g: G
}

impl<G: ActivationFunction> Perceptron<G> {
    fn new(input_size: usize, output_size: usize) -> Self {
        Perceptron {
            weights: PyMatrix::new(input_size, output_size),
            bias: PyVector::new(output_size),
            input_size,
            output_size,
            h: PyVector::new(output_size),
            _g: G::new()
        }
    }
}

impl<G: ActivationFunction> LearningUnit for Perceptron<G> {
    fn load<Reader: BufRead>(reader: &mut Reader) -> Result<Self, Error> {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let input_size: usize = match line.trim().parse() {
            Ok(n) => n,
            Err(e) => {
                return Err(Error::new(std::io::ErrorKind::InvalidData, e));
            }
        };
        line = String::new();
        reader.read_line(&mut line)?;
        let output_size: usize = match line.trim().parse() {
            Ok(n) => n,
            Err(e) => {
                return Err(Error::new(std::io::ErrorKind::InvalidData, e));
            }
        };
        let mut perceptron = Perceptron::new(input_size, output_size);
        for w in perceptron.weights.get_data_mut().iter_mut() {
            line = String::new();
            reader.read_line(&mut line)?;
            *w = match line.trim().parse() {
                Ok(n) => n,
                Err(e) => {
                    return Err(Error::new(std::io::ErrorKind::InvalidData, e));
                }
            };
        }
        for w in perceptron.bias.get_data_mut().iter_mut() {
            line = String::new();
            reader.read_line(&mut line)?;
            *w = match line.trim().parse() {
                Ok(n) => n,
                Err(e) => {
                    return Err(Error::new(std::io::ErrorKind::InvalidData, e));
                }
            };
        }
        Ok(perceptron)
    }

    fn save<Writer: Write>(&self, writer: &mut Writer) -> Result<(), Error> {
        writer.write_all(format!("{}\n", self.input_size).as_bytes())?;
        writer.write_all(format!("{}\n", self.output_size).as_bytes())?;
        for w in self.weights.get_data().iter() {
            writer.write_all(format!("{:.64}\n", *w).as_bytes())?;
        }
        for w in self.bias.get_data().iter() {
            writer.write_all(format!("{:.64}\n", *w).as_bytes())?;
        }
        Ok(())
    }

    fn forward<Vi: Vector, Vo: Vector>(&mut self, input: &Vi, output: &mut Vo) {
        debug_assert_eq!(self.output_size, output.get_data().len());
        debug_assert_eq!(self.input_size, input.get_data().len());
        self.weights.dot(input, &mut self.h);
        self.h.add_assign(&self.bias);
        for i in 0..output.get_data().len() {
            output.get_data_mut()[i] = G::function(self.h.get_data()[i]);
        }
    }

    fn calc_error<Vd: Vector, Ve: Vector>(&mut self, diff: &Vd, error: &mut Ve) {
        debug_assert_eq!(self.output_size, diff.get_data().len());
        debug_assert_eq!(self.output_size, error.get_data().len());
        for (i, e) in error.get_data_mut().iter_mut().enumerate() {
            *e = G::derive(self.h.get_data()[i]) * diff.get_data()[i];
        }
    }

    fn backward_diff<Ve: Vector, Vd: Vector>(&mut self, error: &Ve, diff: &mut Vd) {
        debug_assert_eq!(self.output_size, error.get_data().len());
        debug_assert_eq!(self.input_size, diff.get_data().len());
        self.weights.transpose_dot(error, diff);
    }

    fn update_weights<Vi: Vector, Ve: Vector>(&mut self, input: &Vi, error: &Ve, rate: Float) {
        debug_assert_eq!(self.input_size, input.get_data().len());
        debug_assert_eq!(self.output_size, error.get_data().len());
        let width = self.weights.get_width();
        for y in 0..(self.weights.get_data().len()/width) {
            for x in 0..width {
                self.weights.get_data_mut()[y * width + x] -= input.get_data()[x] * error.get_data()[y] * rate;
            }
        }
        self.bias.add_mul_assign(error, -rate);
    }

    fn get_input_size(&self) -> usize {
        self.input_size
    }

    fn get_output_size(&self) -> usize {
        self.output_size
    }

    fn randomize(&mut self, range: Float) {
        let mut rng = rand::thread_rng();
        for w in self.weights.get_data_mut().iter_mut() {
            *w = rng.gen_range(-range..range);
        }
        for w in self.bias.get_data_mut().iter_mut() {
            *w = rng.gen_range(-range..range);
        }
    }
}

#[pyclass]
struct DataSet{
    #[pyo3(get)]
    input_size: usize,
    #[pyo3(get)]
    label_size: usize,
    #[pyo3(get)]
    inputs: Vec<PyVector>,
    #[pyo3(get)]
    labels: Vec<PyVector>
}

#[pymethods]
impl DataSet {
    #[new]
    fn new(input_size: usize, label_size: usize) -> Self {
        DataSet {
            input_size,
            label_size,
            inputs: Vec::new(),
            labels: Vec::new()
        }
    }

    fn len(&self) -> usize {
        self.inputs.len()
    }

    fn add_entry(&mut self, input: PyVector, label: PyVector) {
        assert_eq!(input.get_data().len(), self.input_size);
        assert_eq!(label.get_data().len(), self.label_size);
        self.inputs.push(input);
        self.labels.push(label);
    }
}

trait Train: LearningUnit {
    fn cost<Vi: Vector, Ve: Vector, Vo: Vector>(&mut self, input: &Vi, expected: &Ve, output: &mut Vo) -> Float;
    fn train<Vi: Vector, Ve: Vector, Vo: Vector>(&mut self, input: &Vi, expected: &Ve, output: &mut Vo, rate: Float) -> Float;
    fn cost_on_set(&mut self, set: &DataSet) -> Float;
    fn train_on_set(&mut self, set: &DataSet, rate: Float) -> Float;
}

impl<L: LearningUnit> Train for L {
    fn cost<Vi: Vector, Ve: Vector, Vo: Vector>(&mut self, input: &Vi, expected: &Ve, output: &mut Vo) -> Float {
        self.forward(input, output);
        let mut diff = output.copy();
        diff.add_mul_assign(expected, -1.0);
        let cost  = diff.dot(&diff);
        cost
    }

    fn train<Vi: Vector, Ve: Vector, Vo: Vector>(&mut self, input: &Vi, expected: &Ve, output: &mut Vo, rate: Float) -> Float {
        self.forward(input, output);
        let mut diff = output.copy();
        diff.add_mul_assign(expected, -1.0);
        let mut error = PyVector::new(output.get_data().len());
        self.calc_error(&diff, &mut error);
        self.update_weights(input, &error, rate);
        let cost  = diff.dot(&diff);
        cost
    }

    fn cost_on_set(&mut self, set: &DataSet) -> Float {
        let mut output = PyVector::new(self.get_output_size());
        let mut cost = 0.0;
        for i in 0..set.len() {
            cost += self.cost(&set.inputs[i], &set.labels[i], &mut output);
        }
        cost/(set.len() as Float)
    }

    fn train_on_set(&mut self, set: &DataSet, rate: Float) -> Float {
        let mut output = PyVector::new(self.get_output_size());
        let mut cost = 0.0;
        for i in 0..set.len() {
            cost += self.train(&set.inputs[i], &set.labels[i], &mut output, rate);
        }
        cost/(set.len() as Float)
    }
}

#[pyclass]
struct PyPerceptron{
    perceptron: Perceptron<ReLU>
}

#[pymethods]
impl PyPerceptron{
    #[new]
    fn new(input_size: usize, output_size: usize) -> PyPerceptron {
        PyPerceptron {
            perceptron: Perceptron::new(input_size, output_size)
        }
    }

    #[staticmethod]
    fn load(path: String) -> PyResult<PyPerceptron> {
        let file = File::open(path)?;
        let perceptron = Perceptron::load(&mut BufReader::new(file))?;
        Ok(PyPerceptron {
            perceptron
        })
    }

    fn save(&mut self, path: String) -> PyResult<()> {
        let file = File::create(path)?;
        let mut buf_writer = BufWriter::new(file);
        self.perceptron.save(&mut buf_writer)?;
        Ok(())
    }

    fn forward(&mut self, input: PyVector) -> PyVector {
        let mut output = PyVector::new(self.perceptron.get_output_size());
        self.perceptron.forward(&input, &mut output);
        output
    }

    fn train(&mut self, input: PyVector, expected: PyVector, rate: Float) -> (PyVector, Float) {
        let mut output = PyVector::new(self.perceptron.get_output_size());
        let cost = self.perceptron.train(&input, &expected, &mut output, rate);
        (output, cost)
    }

    fn train_on_set(&mut self, set: PyRef<DataSet>, rate: Float) -> Float {
        self.perceptron.train_on_set(&(*set), rate)
    }

    fn cost_on_set(&mut self, set: PyRef<DataSet>) -> Float {
        self.perceptron.cost_on_set(&(*set))
    }


    fn randomize(&mut self, range: Float) {
        self.perceptron.randomize(range);
    }
}

#[pymodule]
fn matrix(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyVector>()?;
    m.add_class::<PyMatrix>()?;
    m.add_class::<PyPerceptron>()?;
    m.add_class::<DataSet>()?;
    Ok(())
}
