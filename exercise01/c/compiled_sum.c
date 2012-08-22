// Must be first, per the docs
#include <Python.h>

// for debugging
#include <stdio.h>

#include <numpy/arrayobject.h>

/* This is the core adding function.  Defined as a macro, so we can
   easily define it for multiple types */
#define define_adder(TYPECODE, NUMERIC_TYPE)                            \
  void iadd_ ## TYPECODE ## _ ## TYPECODE (void * rslt,                 \
                                           void * addend) {             \
    *(NUMERIC_TYPE *) rslt += *(NUMERIC_TYPE *) addend;                 \
  }
define_adder( i8,      int8_t)
define_adder(i16,     int16_t)
define_adder(i32,     int32_t)
define_adder(i64,     int64_t)
define_adder( u8,     uint8_t)
define_adder(u16,    uint16_t)
define_adder(u32,    uint32_t)
define_adder(u64,    uint64_t)
define_adder(f32, npy_float32)
define_adder(f64, npy_float64)

/* Compiled function to compute sum
 *
 * Still to do:
 *   * Try releasing the GIL
 *   * Upcast to biggest possible int/unsigned int/float to avoid rollover
 */
static PyObject * sum(PyObject * self, PyObject * args) {
  PyObject * raw_input;
  PyArrayObject * input;
  PyArrayObject * rslt;
  PyArrayIterObject * iter;
  void * rslt_data;
  void * curr_elem;
  void (*adder)(void*,void*);
  int npy_typenum;

  if (!PyArg_ParseTuple(args, "O", &raw_input)) {
    return NULL;
  }

  input = (PyArrayObject *) PyArray_FROM_O(raw_input);

  Py_INCREF(input->descr);
  rslt = (PyArrayObject *) PyArray_Zeros(0, NULL, PyArray_DESCR(input), 0);
  rslt_data = (int64_t *) PyArray_DATA(rslt);

  // Determine output type
  npy_typenum = PyArray_DESCR(rslt)->type_num;
  /* Need to expand the switch statement to handle all the possible
     types (or build up an array of a custom struct and use that, or
     better yet, rewrite as a ufunc. */
  switch(npy_typenum) {
  case (NPY_INT64) :
    adder = iadd_i64_i64;
    break;
  case (NPY_FLOAT64) :
    adder = iadd_f64_f64;
    break;
  default :
    adder = NULL;
    break;
  }
  if (adder == NULL) {
    // Should put the dtype in the error string...
    PyErr_SetString(PyExc_TypeError, "Unable to handle dtype");
    Py_XDECREF(input);
    Py_RETURN_NONE;
  }

  // Now to iterate over the array
  for (iter = (PyArrayIterObject *) PyArray_IterNew((PyObject*)input);
       PyArray_ITER_NOTDONE(iter);) {
    curr_elem = PyArray_ITER_DATA(iter);
    adder(rslt_data, curr_elem);
    PyArray_ITER_NEXT(iter);    
  }

  Py_XDECREF(input);
  return (PyObject*) rslt;
}

static PyMethodDef SumMethods[] = {
  {"sum", sum, METH_VARARGS, "Sum elements of an array"},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initcompiled_sum(void) {
  import_array();
  (void) Py_InitModule("compiled_sum", SumMethods);
}
