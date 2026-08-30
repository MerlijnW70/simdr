use super::{BuildError, Id, Module, Section, op};
use crate::encode::Word;
use crate::spec::StorageClass;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum TypeKey {
    Void,
    Bool,
    Int { width: u32, signed: bool },
    Float { width: u32 },
    Vector { component: Id, count: u32 },
    Pointer { storage: StorageClass, pointee: Id },
    Function { returns: Id, parameters: Vec<Id> },
}

impl Module {
    fn intern_type<F>(&mut self, key: TypeKey, opcode: u16, tail: F) -> Result<Id, BuildError>
    where
        F: FnOnce() -> Vec<Word>,
    {
        if let Some(&existing) = self.types.get(&key) {
            return Ok(existing);
        }

        let id = self.alloc_id()?;
        let mut operands = vec![id.word()];
        operands.extend(tail());
        self.emit(Section::TypeConstantVariable, opcode, &operands)?;
        self.types.insert(key, id);
        Ok(id)
    }

    pub fn type_void(&mut self) -> Result<Id, BuildError> {
        self.intern_type(TypeKey::Void, op::TYPE_VOID, Vec::new)
    }

    pub fn type_bool(&mut self) -> Result<Id, BuildError> {
        self.intern_type(TypeKey::Bool, op::TYPE_BOOL, Vec::new)
    }

    pub fn type_int(&mut self, width: u32, signed: bool) -> Result<Id, BuildError> {
        self.intern_type(TypeKey::Int { width, signed }, op::TYPE_INT, || {
            vec![width, Word::from(signed)]
        })
    }

    pub fn type_float(&mut self, width: u32) -> Result<Id, BuildError> {
        self.intern_type(TypeKey::Float { width }, op::TYPE_FLOAT, || vec![width])
    }

    pub fn type_vector(&mut self, component: Id, count: u32) -> Result<Id, BuildError> {
        self.intern_type(
            TypeKey::Vector { component, count },
            op::TYPE_VECTOR,
            || vec![component.word(), count],
        )
    }

    pub fn type_pointer(&mut self, storage: StorageClass, pointee: Id) -> Result<Id, BuildError> {
        self.intern_type(
            TypeKey::Pointer { storage, pointee },
            op::TYPE_POINTER,
            || vec![storage.word(), pointee.word()],
        )
    }

    pub fn type_function(&mut self, returns: Id, parameters: &[Id]) -> Result<Id, BuildError> {
        let key = TypeKey::Function {
            returns,
            parameters: parameters.to_vec(),
        };
        let tail: Vec<Word> = core::iter::once(returns.word())
            .chain(parameters.iter().map(|parameter| parameter.word()))
            .collect();

        self.intern_type(key, op::TYPE_FUNCTION, || tail)
    }

    pub fn type_runtime_array(&mut self, element: Id) -> Result<Id, BuildError> {
        let id = self.alloc_id()?;
        self.emit(
            Section::TypeConstantVariable,
            op::TYPE_RUNTIME_ARRAY,
            &[id.word(), element.word()],
        )?;
        Ok(id)
    }

    pub fn type_array(&mut self, element: Id, length: Id) -> Result<Id, BuildError> {
        let id = self.alloc_id()?;
        self.emit(
            Section::TypeConstantVariable,
            op::TYPE_ARRAY,
            &[id.word(), element.word(), length.word()],
        )?;
        Ok(id)
    }

    pub fn type_struct(&mut self, members: &[Id]) -> Result<Id, BuildError> {
        let id = self.alloc_id()?;
        let mut operands = vec![id.word()];
        operands.extend(members.iter().map(|member| member.word()));
        self.emit(Section::TypeConstantVariable, op::TYPE_STRUCT, &operands)?;
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::module::Version;

    #[test]
    fn asking_for_the_same_type_twice_declares_it_once() {
        let mut module = Module::new(Version::V1_3);

        let first = module.type_int(32, false).expect("u32");
        let second = module.type_int(32, false).expect("u32 again");

        assert_eq!(first, second, "the same shape is the same type");
        assert_eq!(
            module.finish().len(),
            5 + 4,
            "one OpTypeInt of four words, not two"
        );
    }

    #[test]
    fn signedness_makes_two_types_out_of_one_width() {
        let mut module = Module::new(Version::V1_3);

        let unsigned = module.type_int(32, false).expect("u32");
        let signed = module.type_int(32, true).expect("i32");

        assert_ne!(unsigned, signed);
    }

    #[test]
    fn width_makes_two_types_out_of_one_signedness() {
        let mut module = Module::new(Version::V1_3);

        let narrow = module.type_int(16, true).expect("i16");
        let wide = module.type_int(32, true).expect("i32");

        assert_ne!(narrow, wide);
    }

    #[test]
    fn a_vectors_component_type_is_part_of_its_identity() {
        let mut module = Module::new(Version::V1_3);
        let float = module.type_float(32).expect("f32");
        let int = module.type_int(32, true).expect("i32");

        let of_floats = module.type_vector(float, 4).expect("vec4");
        let of_ints = module.type_vector(int, 4).expect("ivec4");
        let again = module.type_vector(float, 4).expect("vec4 again");

        assert_ne!(of_floats, of_ints);
        assert_eq!(of_floats, again);
    }

    #[test]
    fn a_vectors_component_count_is_part_of_its_identity() {
        let mut module = Module::new(Version::V1_3);
        let float = module.type_float(32).expect("f32");

        let pair = module.type_vector(float, 2).expect("vec2");
        let quad = module.type_vector(float, 4).expect("vec4");

        assert_ne!(pair, quad);
    }

    #[test]
    fn a_function_types_parameter_list_is_part_of_its_identity() {
        let mut module = Module::new(Version::V1_3);
        let void = module.type_void().expect("void");
        let int = module.type_int(32, true).expect("i32");

        let takes_nothing = module.type_function(void, &[]).expect("fn()");
        let takes_one = module.type_function(void, &[int]).expect("fn(i32)");
        let takes_two = module
            .type_function(void, &[int, int])
            .expect("fn(i32,i32)");
        let again = module.type_function(void, &[int]).expect("fn(i32) again");

        assert_ne!(takes_nothing, takes_one);
        assert_ne!(takes_one, takes_two);
        assert_eq!(takes_one, again);
    }

    #[test]
    fn a_function_types_return_type_is_part_of_its_identity() {
        let mut module = Module::new(Version::V1_3);
        let void = module.type_void().expect("void");
        let int = module.type_int(32, true).expect("i32");

        let returns_nothing = module.type_function(void, &[]).expect("fn() -> ()");
        let returns_int = module.type_function(int, &[]).expect("fn() -> i32");

        assert_ne!(returns_nothing, returns_int);
    }

    #[test]
    fn a_pointers_storage_class_is_part_of_its_identity() {
        let mut module = Module::new(Version::V1_3);
        let float = module.type_float(32).expect("f32");

        let in_buffer = module
            .type_pointer(StorageClass::StorageBuffer, float)
            .expect("buffer pointer");
        let in_workgroup = module
            .type_pointer(StorageClass::Workgroup, float)
            .expect("workgroup pointer");

        assert_ne!(in_buffer, in_workgroup);
    }

    #[test]
    fn a_type_is_declared_before_anything_that_uses_it() {
        let mut module = Module::new(Version::V1_3);
        let float = module.type_float(32).expect("f32");
        module.type_vector(float, 4).expect("vec4");

        let words = module.finish();
        let body = &words[5..];

        assert_eq!(body[0] & 0xffff, Word::from(op::TYPE_FLOAT));
        assert_eq!(body[3] & 0xffff, Word::from(op::TYPE_VECTOR));
    }

    #[test]
    fn an_array_type_names_its_element_and_its_length_and_a_runtime_one_names_no_length() {
        let mut module = Module::new(Version::V1_3);
        let float = module.type_float(32).expect("f32");
        let length = module.constant_u32(4).expect("4");

        let sized = module.type_array(float, length).expect("array");
        let unsized_array = module.type_runtime_array(float).expect("runtime array");

        let words = module.finish();
        let operands = |opcode: u16| {
            crate::decode::body(&words)
                .find(|instruction| instruction.opcode() == opcode)
                .expect("the type was declared")
                .operands()
                .to_vec()
        };

        assert_eq!(
            operands(op::TYPE_ARRAY),
            vec![sized.word(), float.word(), length.word()],
            "the length is an id and not a literal, which is the whole difference between \
             OpTypeArray and a size somebody wrote down"
        );
        assert_eq!(
            operands(op::TYPE_RUNTIME_ARRAY),
            vec![unsized_array.word(), float.word()],
            "a runtime array has no length to carry"
        );
    }

    #[test]
    fn two_arrays_of_the_same_shape_are_two_types_the_way_two_structs_are() {
        let mut module = Module::new(Version::V1_3);
        let float = module.type_float(32).expect("f32");
        let length = module.constant_u32(4).expect("4");

        let first = module.type_array(float, length).expect("array");
        let second = module.type_array(float, length).expect("array again");
        let first_runtime = module.type_runtime_array(float).expect("runtime array");
        let second_runtime = module
            .type_runtime_array(float)
            .expect("runtime array again");

        assert_ne!(first, second);
        assert_ne!(first_runtime, second_runtime);
    }
}
