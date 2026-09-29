#![cfg_attr(not(feature = "std"), no_std)]
use frame_support::pallet_prelude::*;
use frame_support::traits::OneSessionHandler;
use frame_system::offchain::SubmitTransaction;

use codec::{Encode, Decode, MaxEncodedLen};

use sp_core::offchain::StorageKind;
use sp_io::hashing::blake2_256;
use sp_runtime::{
	BoundedSlice, KeyTypeId, RuntimeAppPublic, traits::{Convert, Extrinsic, Member, OpaqueKeys, Zero}
};
use sp_staking::SessionIndex;
use sp_std::{
	marker::PhantomData,
	ops::{Rem, Sub},
	prelude::*,
};

pub mod historical;

pub use pallet::*;

pub const LOG_TARGET: &str = "runtime::guard-session";

/// Domain separator for agreement response signatures.
pub const AGREEMENT_RESPONSE_CONTEXT: &[u8] = b"palliora/guard-agreement-response/v1";

mod app {
	use scale_info::prelude::string::String;
	use sp_application_crypto::{app_crypto, key_types::GUARDIAN, sr25519};
	app_crypto!(sr25519, GUARDIAN);
}

sp_application_crypto::with_pair! {
	/// An authority discovery authority keypair.
	pub type GuardianPair = app::Pair;
}

/// An authority discovery authority identifier.
pub type GuardianId = app::Public;

/// An authority discovery authority signature.
pub type GuardianSignature = app::Signature;

impl<T: Config> sp_runtime::BoundToRuntimeAppPublic for Pallet<T> {
	type Public = GuardianId;
}

/// Decides whether the session should be ended.
pub trait ShouldEndSession<BlockNumber> {
	/// Return `true` if the session should be ended.
	fn should_end_session(now: BlockNumber) -> bool;
}

/// Ends the session after a fixed period of blocks.
///
/// The first session will have length of `Offset`, and
/// the following sessions will have length of `Period`.
/// This may prove nonsensical if `Offset` >= `Period`.
pub struct PeriodicSessions<Period, Offset>(PhantomData<(Period, Offset)>);

impl<
		BlockNumber: Rem<Output = BlockNumber> + Sub<Output = BlockNumber> + Zero + PartialOrd,
		Period: Get<BlockNumber>,
		Offset: Get<BlockNumber>,
	> ShouldEndSession<BlockNumber> for PeriodicSessions<Period, Offset>
{
	fn should_end_session(now: BlockNumber) -> bool {
		let offset = Offset::get();
		now >= offset && ((now - offset) % Period::get()).is_zero()
	}
}

/// Error which may occur while executing the off-chain code.
#[cfg_attr(test, derive(PartialEq))]
#[derive(Debug)]
enum OffchainErr {
	TooEarly,
	AlreadyOnline(u32),
	FailedSigning,
	FailedToAcquireLock,
	SubmitTransaction,
	/// The account has no guardian id, so no registered key can be looked up for it.
	UnknownGuardian,
	/// No `GUARDIAN` session key is registered on chain, or its secret is not in the keystore.
	MissingGuardKey,
}

type OffchainResult<A> = Result<A, OffchainErr>;

#[derive(Default, Decode, Encode, PartialEq)]
pub enum AgreementAction {
	#[default]
	Ignore,
	Accept,
	Reject,
}

#[derive(Default, Decode, Encode, PartialEq)]
pub enum AgreementState {
	#[default]
	Pending,
	Processed,
}

#[derive(Default, Decode, Encode)]
pub struct AgreementEntry(AgreementAction, AgreementState, [u8; 32]);

/// A trait for managing creation of new guardian set.
pub trait SessionManager<GuardianId> {
	fn test(_: SessionIndex) {}
	/// Plan a new session, and optionally provide the new guardian set.
	///
	/// Even if the guardian-set is the same as before, if any underlying economic conditions have
	/// changed (i.e. stake-weights), the new guardian set must be returned. This is necessary for
	/// consensus engines making use of the session pallet to issue a guardian-set change so
	/// misbehavior can be provably associated with the new economic conditions as opposed to the
	/// old. The returned guardian set, if any, will not be applied until `new_index`. `new_index`
	/// is strictly greater than from previous call.
	///
	/// The first session start at index 0.
	///
	/// `new_session(session)` is guaranteed to be called before `end_session(session-1)`. In other
	/// words, a new session must always be planned before an ongoing one can be finished.
	fn new_session(new_index: SessionIndex) -> Option<Vec<GuardianId>>;
	/// Same as `new_session`, but it this should only be called at genesis.
	///
	/// The session manager might decide to treat this in a different way. Default impl is simply
	/// using [`new_session`](Self::new_session).
	fn new_session_genesis(new_index: SessionIndex) -> Option<Vec<GuardianId>> {
		Self::new_session(new_index)
	}
	/// End the session.
	///
	/// Because the session pallet can queue guardian set the ending session can be lower than the
	/// last new session index.
	fn end_session(end_index: SessionIndex);
	/// Start an already planned session.
	///
	/// The session start to be used for validation.
	fn start_session(start_index: SessionIndex);
}

impl<A> SessionManager<A> for () {
	fn test(_: SessionIndex) {}
	fn new_session(_: SessionIndex) -> Option<Vec<A>> {
		None
	}
	fn start_session(_: SessionIndex) {}
	fn end_session(_: SessionIndex) {}
}

/// Implementors of this trait provide information about whether or not some guardian has
/// been registered with them. The [Session module](../../pallet_session/index.html) is an
/// implementor.
pub trait GuardianRegistration<GuardianId> {
	/// Returns true if the provided guardian ID has been registered with the implementing runtime
	/// module
	fn is_registered(id: &GuardianId) -> bool;

	/// The `GUARDIAN` session key registered for `id`, if any.
	///
	/// This is the public half of the key a guardian signs agreement responses with, and the
	/// only on-chain binding from a guardian account to signable material. `WorkerByKey` holds
	/// the node's libp2p network key instead, whose secret never reaches the keystore.
	fn guard_key(id: &GuardianId) -> Option<crate::GuardianId>;
}

/// Handler for session life cycle events.
pub trait SessionHandler<GuardianId> {
	/// All the key type ids this session handler can process.
	///
	/// The order must be the same as it expects them in
	/// [`on_new_session`](Self::on_new_session<Ks>) and
	/// [`on_genesis_session`](Self::on_genesis_session<Ks>).
	const KEY_TYPE_IDS: &'static [KeyTypeId];

	/// The given guardian set will be used for the genesis session.
	/// It is guaranteed that the given guardian set will also be used
	/// for the second session, therefore the first call to `on_new_session`
	/// should provide the same guardian set.
	fn on_genesis_session<Ks: OpaqueKeys>(guardians: &[(GuardianId, Ks)]);

	/// Session set has changed; act appropriately. Note that this can be called
	/// before initialization of your pallet.
	///
	/// `changed` is true whenever any of the session keys or underlying economic
	/// identities or weightings behind those keys has changed.
	fn on_new_session<Ks: OpaqueKeys>(
		changed: bool,
		guardians: &[(GuardianId, Ks)],
		queued_guardians: &[(GuardianId, Ks)],
	);

	/// A notification for end of the session.
	///
	/// Note it is triggered before any [`SessionManager::end_session`] handlers,
	/// so we can still affect the guardian set.
	fn on_before_session_ending() {}

	/// A guardian got disabled. Act accordingly until a new session begins.
	fn on_disabled(guardian_index: u32);
}

#[impl_trait_for_tuples::impl_for_tuples(1, 30)]
#[tuple_types_custom_trait_bound(OneSessionHandler<AId>)]
impl<AId> SessionHandler<AId> for Tuple {
	for_tuples!(
		const KEY_TYPE_IDS: &'static [KeyTypeId] = &[ #( <Tuple::Key as RuntimeAppPublic>::ID ),* ];
	);

	fn on_genesis_session<Ks: OpaqueKeys>(guardians: &[(AId, Ks)]) {
		// NOTE: Disabled as it currently causes issues with the macro expansion.
		log::warn!(target: LOG_TARGET, "OneSessionHandler on_genesis_session macro implementation is disabled due to macro expansion issues.");
		// for_tuples!(
		// 	#(
		// 		let our_keys: Box<dyn Iterator<Item=_>> = Box::new(guardians.iter()
		// 			.filter_map(|k|
		// 				k.1.get::<Tuple::Key>(<Tuple::Key as RuntimeAppPublic>::ID).map(|k1| (&k.0, k1))
		// 			)
		// 		);

		// 		Tuple::on_genesis_session(our_keys);
		// 	)*
		// )
	}

	fn on_new_session<Ks: OpaqueKeys>(
		changed: bool,
		guardians: &[(AId, Ks)],
		queued_guardians: &[(AId, Ks)],
	) {
		// NOTE: Disabled as it currently causes issues with the macro expansion.
		log::warn!(target: LOG_TARGET, "OneSessionHandler on_new_session macro implementation is disabled due to macro expansion issues.");
		// for_tuples!(
		// 	#(
		// 		let our_keys: Box<dyn Iterator<Item=_>> = Box::new(guardians.iter()
		// 			.filter_map(|k|
		// 				k.1.get::<Tuple::Key>(<Tuple::Key as RuntimeAppPublic>::ID).map(|k1| (&k.0, k1))
		// 			));
		// 		let queued_keys: Box<dyn Iterator<Item=_>> = Box::new(queued_guardians.iter()
		// 			.filter_map(|k|
		// 				k.1.get::<Tuple::Key>(<Tuple::Key as RuntimeAppPublic>::ID).map(|k1| (&k.0, k1))
		// 			));
		// 		Tuple::on_new_session(changed, our_keys, queued_keys);
		// 	)*
		// )
	}

	fn on_before_session_ending() {
		for_tuples!( #( Tuple::on_before_session_ending(); )* )
	}

	fn on_disabled(i: u32) {
		for_tuples!( #( Tuple::on_disabled(i); )* )
	}
}


/// `SessionHandler` for tests that use `UintAuthorityId` as `Keys`.
pub struct TestSessionHandler;
impl<AId> SessionHandler<AId> for TestSessionHandler {
	const KEY_TYPE_IDS: &'static [KeyTypeId] = &[sp_runtime::key_types::DUMMY];
	fn on_genesis_session<Ks: OpaqueKeys>(_: &[(AId, Ks)]) {}
	fn on_new_session<Ks: OpaqueKeys>(_: bool, _: &[(AId, Ks)], _: &[(AId, Ks)]) {}
	fn on_before_session_ending() {}
	fn on_disabled(_: u32) {}
}

#[frame_support::pallet]
pub mod pallet {

use super::*;
	
	use frame_system::{offchain::SendTransactionTypes, pallet_prelude::*};

	

	/// The current storage version.
	const STORAGE_VERSION: StorageVersion = StorageVersion::new(0);

	#[pallet::pallet]
	#[pallet::storage_version(STORAGE_VERSION)]
	#[pallet::without_storage_info]
	pub struct Pallet<T>(_);

	#[pallet::config]
	pub trait Config: SendTransactionTypes<Call<Self>> + frame_system::Config {
		/// The overarching event type.
		type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;

		/// A stable ID for a guardian.
		type GuardianId: Member
			+ Parameter
			+ MaybeSerializeDeserialize
			+ MaxEncodedLen
			+ TryFrom<Self::AccountId>;

		/// A conversion from account ID to guardian ID.
		///
		/// Its cost must be at most one storage read.
		type ValidatorIdOf: Convert<Self::AccountId, Option<Self::GuardianId>>;

		/// Indicator for when to end the session.
		type ShouldEndSession: ShouldEndSession<BlockNumberFor<Self>>;

		/// Handler for managing new session.
		type SessionManager: SessionManager<Self::GuardianId>;

		/// Validate a guardian's registration status.
		type GuardianRegistration: GuardianRegistration<Self::GuardianId>;

		/// The maximum number of keys that can be added.
		type MaxKeys: Get<u32>;
	}

	#[pallet::genesis_config]
	#[derive(frame_support::DefaultNoBound)]
	pub struct GenesisConfig<T: Config> {
		pub _phantom: PhantomData<T>,
	}

	#[pallet::genesis_build]
	impl<T: Config> BuildGenesisConfig for GenesisConfig<T> {
		fn build(&self) {
			let initial_guardians_0 =
				T::SessionManager::new_session_genesis(0).unwrap_or_else(|| {
					frame_support::print(
						"No initial guardian provided by `SessionManager`, use \
						session config keys to generate initial guardian set.",
					);
					Vec::new()
				});
			// assert!(
			// 	!initial_guardians_0.is_empty(),
			// 	"Empty guardian set for session 0 in genesis block!"
			// );

			let initial_guardians_1 = T::SessionManager::new_session_genesis(1)
				.unwrap_or_else(|| initial_guardians_0.clone());
			// assert!(
			// 	!initial_guardians_1.is_empty(),
			// 	"Empty guardian set for session 1 in genesis block!"
			// );

			// Tell everyone about the genesis session keys
			// T::SessionHandler::on_genesis_session::<T::Keys>(&queued_keys);

			Guardians::<T>::put(initial_guardians_0);

			T::SessionManager::start_session(0);
		}
	}

	/// The current set of guardians.
	#[pallet::storage]
	#[pallet::getter(fn guardians)]
	pub type Guardians<T: Config> = StorageValue<_, Vec<T::GuardianId>, ValueQuery>;


	/// Mapping from guardian session key to stash AccountId.
	/// Used by the offchain worker to resolve its own AccountId without needing the peer id.
	#[pallet::storage]
	#[pallet::getter(fn worker_by_key)]
	pub type WorkerByKey<T: Config> =
			StorageMap<_, Twox64Concat, GuardianId, T::AccountId, OptionQuery>;

	/// Reverse mapping from stash AccountId to guardian session key.
	#[pallet::storage]
	#[pallet::getter(fn worker_by_account)]
	pub type WorkerByAccount<T: Config> =
		StorageMap<_, Twox64Concat, T::AccountId, GuardianId, OptionQuery>;

	/// The upcoming (next) set of guardians.
	#[pallet::storage]
	#[pallet::getter(fn next_guardians)]
	pub type NextGuardians<T: Config> = StorageValue<_, Vec<T::GuardianId>, ValueQuery>;

	/// True if the underlying economic identities or weighting behind the validators
	/// has changed in the queued validator set.
	#[pallet::storage]
	pub type QueuedChanged<T> = StorageValue<_, bool, ValueQuery>;

	/// Current index of the session.
	#[pallet::storage]
	#[pallet::getter(fn current_index)]
	pub type CurrentIndex<T> = StorageValue<_, SessionIndex, ValueQuery>;

	/// Indices of disabled guardians.
	///
	/// The vec is always kept sorted so that we can find whether a given guardian is
	/// disabled using binary search. It gets cleared when `on_session_ending` returns
	/// a new set of identities.
	#[pallet::storage]
	#[pallet::getter(fn disabled_guardians)]
	pub type DisabledGuardians<T> = StorageValue<_, Vec<u32>, ValueQuery>;

	#[pallet::storage]
	#[pallet::getter(fn default_groups_marker)]
	pub type DefafaultGroupsMarker<T> = StorageValue<_, bool, ValueQuery>;

	/// The current set of keys that may issue a heartbeat.
	#[pallet::storage]
	#[pallet::getter(fn keys)]
	pub(super) type Keys<T: Config> =
		StorageValue<_, WeakBoundedVec<GuardianId, T::MaxKeys>, ValueQuery>;

	#[pallet::storage]
	#[pallet::getter(fn agreement_responses)]
	pub type AgreementsResponses<T: Config> = StorageMap<_, Twox64Concat, [u8; 32], Vec<(T::AccountId, bool)>, OptionQuery>;

	#[pallet::event]
	#[pallet::generate_deposit(pub(super) fn deposit_event)]
	pub enum Event<T: Config> {
		/// New session has happened. Note that the argument is the session index, not the
		/// block number as the type might suggest.
		NewSession { session_index: SessionIndex },
		AgreementResponse { agrement: [u8; 32], signer: T::AccountId, acceptance: bool },
	}

	#[pallet::error]
	pub enum Error<T> {
		/// Dummy error.
		Dummy,
	}

	#[pallet::hooks]
	impl<T: Config> Hooks<BlockNumberFor<T>> for Pallet<T> {
		/// Ref: https://github.com/paritytech/polkadot-sdk/blob/935c7f461ae8b4e607f1db16322ea952b438650e/substrate/frame/examples/offchain-worker/src/lib.rs#L539
		///
		/// Note that it's not guaranteed for offchain workers to run on EVERY block, there might
		/// be cases where some blocks are skipped, or for some the worker runs twice (re-orgs),
		/// so the code should be able to handle that.
		/// You can use `Local Storage` API to coordinate runs of the worker.
		fn on_initialize(n: BlockNumberFor<T>) -> Weight {
			log::warn!(target: LOG_TARGET, "on_initialize {:?}", n);
			if T::ShouldEndSession::should_end_session(n) {
				log::warn!(target: LOG_TARGET, "ending session {:?}", n);
				Self::rotate_guard_session();
				T::BlockWeights::get().max_block
			} else {
				// NOTE: the non-database part of the weight for `should_end_session(n)` is
				// included as weight for empty block, the database part is expected to be in
				// cache.
				Weight::zero()
			}
		}

		fn offchain_worker(now: BlockNumberFor<T>) {
			// Only send messages if we are a potential validator.
			if sp_io::offchain::is_guardian() {
				if let Err(e) = Self::accept_agreements() {
					log::warn!(target: LOG_TARGET, "Failed to respond to agreements: {e:?}");
				}
			} else {
				log::trace!(
					target: LOG_TARGET,
					"Accepting agreements at {:?}. Not a validator.",
					now,
				)
			}
		}
	}

	#[pallet::call]
	impl<T: Config> Pallet<T> {
		/// Take the origin account as a stash and lock up `value` of its balance. `controller` will
		/// be the account that controls it.
		///
		/// `value` must be more than the `minimum_balance` specified by `T::Currency`.
		///
		/// The dispatch origin for this call must be _Signed_ by the stash account.
		///
		/// Emits `Bonded`.
		/// ## Complexity
		/// - Independent of the arguments. Moderate complexity.
		/// - O(1).
		/// - Three extra DB entries.
		///
		/// NOTE: Two of the storage writes (`Self::bonded`, `Self::payee`) are _never_ cleaned
		/// unless the `origin` falls below _existential deposit_ and gets removed as dust.
		#[pallet::call_index(0)]
		#[pallet::weight(Weight::from_parts(16_980_000, 4556))]
		pub fn set_worker(
			origin: OriginFor<T>,
			guardian_key: GuardianId,
		) -> DispatchResult {
			let who = ensure_signed(origin)?;
			if let Some(old_key) = <WorkerByAccount<T>>::get(&who) {
				<WorkerByKey<T>>::remove(old_key);
			}
			<WorkerByKey<T>>::insert(guardian_key.clone(), who.clone());
			<WorkerByAccount<T>>::insert(who, guardian_key);

			Ok(())
		}

		#[pallet::call_index(1)]
		#[pallet::weight(Weight::from_parts(16_980_000, 4556))]
		pub fn agreement_response(
			origin: OriginFor<T>,
			agreementid: [u8; 32],
			address: T::AccountId,
			// since signature verification is done in `validate_unsigned`
			// we can skip doing it here again.
			_signature: <GuardianId as RuntimeAppPublic>::Signature,
			acceptance: bool,
		) -> DispatchResultWithPostInfo {
			ensure_none(origin)?;
			AgreementsResponses::<T>::mutate(&agreementid, |opt| {
				opt.get_or_insert_with(Vec::new).push((address.clone(), acceptance));
			});
			Self::deposit_event(Event::AgreementResponse { agrement: agreementid, signer: address, acceptance});

			Ok(().into())
		}
	}

	/// Invalid transaction custom error. Returned when validators_len field in heartbeat is
	/// incorrect.
	pub(crate) const INVALID_VALIDATORS_LEN: u8 = 10;

	#[pallet::validate_unsigned]
	impl<T: Config> ValidateUnsigned for Pallet<T> {
		type Call = Call<T>;

		fn validate_unsigned(_source: TransactionSource, call: &Self::Call) -> TransactionValidity {
			if let Call::agreement_response { agreementid, address, signature, acceptance } = call {
				// `address` is the guardian's own id, as `rotate_guard_session` also assumes
				// when it hands ids straight to `is_registered`. Converting through
				// `ValidatorIdOf` would instead demand a staking ledger keyed by controller.
				let guardian_id = T::GuardianId::try_from(address.clone())
					.map_err(|_| InvalidTransaction::BadSigner)?;
				let guard_key = T::GuardianRegistration::guard_key(&guardian_id)
					.ok_or(InvalidTransaction::BadSigner)?;

				let payload = Self::agreement_payload(agreementid, address, *acceptance);
				if !guard_key.verify(&payload, signature) {
					return InvalidTransaction::BadProof.into();
				}

				ValidTransaction::with_tag_prefix("GuardAgreement")
					.priority(TransactionPriority::MAX)
					.and_provides((address, agreementid))
					.longevity(64_u64)
					.propagate(true)
					.build()
			} else {
				InvalidTransaction::Call.into()
			}
		}
	}
}

impl<T: Config> Pallet<T> {
	/// Move on to next session. Register new guardian set and session keys. Changes to the
	/// guardian set have a session of delay to take effect. This allows for equivocation
	/// punishment after a fork.
	pub fn rotate_guard_session() {
		let changed = QueuedChanged::<T>::get();
		let session_index = <CurrentIndex<T>>::get();
		log::info!(target: LOG_TARGET, "rotating guard session {:?}", session_index);
		T::SessionManager::test(session_index);

		// Inform the session handlers that a session is going to end.
		// T::SessionHandler::on_before_session_ending();
		T::SessionManager::end_session(session_index);

		// Get queued session keys and guardians.
		let next_guardians = NextGuardians::<T>::get();
		Guardians::<T>::put(&next_guardians);

		if changed {
			// reset disabled guardians
			<DisabledGuardians<T>>::take();
		}

		// Increment session index.
		let session_index = session_index + 1;
		<CurrentIndex<T>>::put(session_index);
		T::SessionManager::test(session_index);

		T::SessionManager::start_session(session_index);
		T::SessionManager::test(session_index);

		// Get next guardian set.
		let maybe_next_guardians = T::SessionManager::new_session(session_index + 1);
		T::SessionManager::test(session_index + 1);
		log::info!(
			target: LOG_TARGET,
			"Next guardian list for session {}: {:?}",
			session_index + 1,
			maybe_next_guardians
		);
		let (next_guardians, next_identities_changed) =
			if let Some(guardians) = maybe_next_guardians.clone() {
				// NOTE: as per the documentation on `OnSessionEnding`, we consider
				// the guardian set as having changed even if the guardians are the
				// same as before, as underlying economic conditions may have changed.
				(guardians, true)
			} else {
				(Guardians::<T>::get(), false)
			};
		// Filter out any guardian that has no session keys returned by `load_keys`.
		let next_guardians: Vec<_> = next_guardians
			.into_iter()
			.filter_map(|g| {
				let loaded = T::GuardianRegistration::is_registered(&g);
				// treat empty result as "no keys" and discard the guardian
				if !loaded {
					log::warn!(target: LOG_TARGET, "discarding guardian with no session keys: {:?}", g);
					None
				} else {
					Some(g)
				}
			})
			.collect();
		log::info!(
			target: LOG_TARGET,
			"Next guardian set for session {}: {:?}",
			session_index + 1,
			next_guardians
		);

		let prev_queued = NextGuardians::<T>::get();
		let mut queued_changed_flag = false;

		// If lengths differ it's a change; otherwise check for any element differences.
		if prev_queued.len() != next_guardians.len() {
			queued_changed_flag = true;
		} else {
			for g in &next_guardians {
				if !prev_queued.iter().any(|p| p == g) {
					queued_changed_flag = true;
					break;
				}
			}
		}

		DefafaultGroupsMarker::<T>::put(maybe_next_guardians.is_some() && !queued_changed_flag);
		QueuedChanged::<T>::put(maybe_next_guardians.is_some() && queued_changed_flag);
		NextGuardians::<T>::put(&next_guardians);

		// Record that this happened.
		Self::deposit_event(Event::NewSession { session_index });

		// Tell everyone about the new session keys.
		// T::SessionHandler::on_new_session::<T::Keys>(changed, &session_keys, &queued_amalgamated);
	}

	pub fn set_groups_marker(value: bool) {
		DefafaultGroupsMarker::<T>::put(value);
	}

	/// The bytes a guardian signs to endorse an agreement response.
	///
	/// The address and the acceptance bit are bound alongside the agreement so a signature
	/// cannot be re-aimed at another guardian or flipped from accept to reject; the genesis
	/// hash and the context string keep it off other chains and other calls.
	pub fn agreement_payload(
		agreementid: &[u8; 32],
		address: &T::AccountId,
		acceptance: bool,
	) -> Vec<u8> {
		let genesis = frame_system::Pallet::<T>::block_hash(
			frame_system::pallet_prelude::BlockNumberFor::<T>::zero(),
		);
		(AGREEMENT_RESPONSE_CONTEXT, genesis, agreementid, address, acceptance).encode()
	}

	fn sign_and_send(address: T::AccountId, agreements: Vec<([u8; 32], bool)>) -> OffchainResult<()> {
		let guardian_id =
			T::GuardianId::try_from(address.clone()).map_err(|_| OffchainErr::UnknownGuardian)?;
		let registered =
			T::GuardianRegistration::guard_key(&guardian_id).ok_or(OffchainErr::MissingGuardKey)?;

		// Sign with the key the chain has registered for this account rather than with whatever
		// `GUARDIAN` keys happen to be in the keystore: only this one verifies on the far side.
		let key = GuardianId::all()
			.into_iter()
			.find(|local| local == &registered)
			.ok_or(OffchainErr::MissingGuardKey)?;

		log::info!(target: LOG_TARGET, "Signing agreements as guardian {registered:?}");

		for (agreementid, acceptance) in agreements {
			let payload = Self::agreement_payload(&agreementid, &address, acceptance);
			let signature = key.sign(&payload).ok_or(OffchainErr::FailedSigning)?;

			let call = Call::agreement_response {
				agreementid,
				address: address.clone(),
				signature,
				acceptance,
			};

			SubmitTransaction::<T, Call<T>>::submit_unsigned_transaction(call.into())
				.unwrap_or_else(|e| {
					log::error!(target: LOG_TARGET, "Failed to submit agreement transaction. Error = {e:?}");
				});
		}

		Ok(())
	}

	pub fn accept_agreements() -> OffchainResult<()> {
		let lst = sp_io::offchain::future_transactions().unwrap_or_default();
		let mut agreements = Vec::new();

		// Derive the guardian key from the local network PeerId (last 32 bytes),
		// consistent with how peer_available() and local_guardian_account() work.
		let nw_state = sp_io::offchain::network_state()
			.map_err(|_| OffchainErr::FailedToAcquireLock)?;
		let raw = nw_state.peer_id.0;
		let Ok(guardian_key): Result<[u8; 32], _> =
			raw[raw.len().saturating_sub(32)..].try_into()
		else {
			log::warn!(
				target: LOG_TARGET,
				"PeerId too short to derive guardian key; skipping agreements"
			);
			return Ok(());
		};
		use sp_core::crypto::UncheckedFrom;
		let my_address: T::AccountId =
			match WorkerByKey::<T>::get(GuardianId::unchecked_from(guardian_key)) {
				Some(addr) => addr,
				None => {
					log::warn!(
						target: LOG_TARGET,
						"No registered address found for local PeerId-derived guardian key; skipping agreements"
					);
					return Ok(());
				},
			};

		for item in lst.into_iter() {
			let item_hash = blake2_256(&item);
			let store = sp_io::offchain::local_storage_get(StorageKind::PERSISTENT, item_hash.as_ref());

			if store.is_none() {
				log::trace!(target: LOG_TARGET, "Skipping filtered agreement");
				continue;
			}

			let store = store.unwrap();
			let et = AgreementEntry::decode(&mut store.as_slice()).unwrap_or(AgreementEntry::default());

			if AgreementAction::Ignore == et.0 || AgreementState::Processed == et.1 {
				log::trace!(target: LOG_TARGET, "Skipping processed agreement");
				continue;
			}

			log::trace!(target: LOG_TARGET, "Accepting agreement {:?}", et.2);
			agreements.push((et.2, AgreementAction::Accept == et.0));

			sp_io::offchain::local_storage_set(StorageKind::PERSISTENT, item_hash.as_ref(), Encode::encode(&(et.0, AgreementState::Processed, et.2)).as_slice());
		}

		log::info!(target: LOG_TARGET, "agreements = {agreements:?}");
		Self::sign_and_send(my_address, agreements)
	}

	fn initialize_keys(keys: &[GuardianId]) {
		if !keys.is_empty() {
			assert!(Keys::<T>::get().is_empty(), "Keys are already initialized!");
			let bounded_keys = <BoundedSlice<'_, _, T::MaxKeys>>::try_from(keys)
				.expect("More than the maximum number of keys provided");
			Keys::<T>::put(bounded_keys);
		}
	}

	#[cfg(test)]
	fn set_keys(keys: Vec<T::GuardianId>) {
		let bounded_keys = WeakBoundedVec::<_, T::MaxKeys>::try_from(keys)
			.expect("More than the maximum number of keys provided");
		Keys::<T>::put(bounded_keys);
	}
}

impl<T: Config> OneSessionHandler<T::GuardianId> for Pallet<T> {
	type Key = GuardianId;

	fn on_before_session_ending() {
		let keys = Keys::<T>::get();

		log::warn!(
			target: LOG_TARGET,
			"Session ended. Validators: {:?}",
			keys,
		);
	}

	fn on_disabled(_validator_index: u32) {
		// ignore
	}

	fn on_genesis_session<'a, I: 'a>(validators: I)
	where
		I: Iterator<Item = (&'a T::GuardianId, Self::Key)>,
		T::GuardianId: 'a 
	{
		let keys = validators.map(|x| x.1).collect::<Vec<_>>();
		Self::initialize_keys(&keys);
		log::warn!(target: LOG_TARGET, "on_genesis_session");
	}

	fn on_new_session<'a, I: 'a>(changed: bool, validators: I, queued_validators: I)
	where
		I: Iterator<Item = (&'a T::GuardianId, Self::Key)>,
		T::GuardianId: 'a
	{
		// Remember who the authorities are for the new session.
		let keys = validators.map(|x| {
			log::info!(target: LOG_TARGET, "validators of the session = {x:?}");
			x.1
		}).collect::<Vec<_>>();
		let bounded_keys = WeakBoundedVec::<_, T::MaxKeys>::force_from(
			keys,
			Some(
				"Warning: The session has more keys than expected. \
				A runtime configuration adjustment may be needed.",
			),
		);
		Keys::<T>::put(bounded_keys);
		log::warn!(target: LOG_TARGET, "on_new_session");
	}
}
