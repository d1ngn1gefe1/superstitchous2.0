#pragma once

#include <list>
#include <utility>
#include <map>

using namespace std;

// evict least-recently-used cache
template <class Key, class Val>
struct LRUCache {
	typedef pair<Key, Val> ListElem;

	size_t sz;
	list<ListElem> l;
	map<Key, typename list<ListElem>::iterator> m;

	LRUCache(size_t sz)
			: sz(sz) {
	}

	// getVal should have signature: void getVal(Key key, Val &out);
	template <class Func>
	void get(const Key &key, Val &out, Func getVal) {
		if (m.count(key)) {
			auto it = m[key];
			out = it->second;
			l.push_front(*it);
			m[key] = l.begin();
			l.erase(it);
			return;
		}

		if (l.size() == sz) {
			m.erase(l.back().first);
			l.pop_back();
		}

		getVal(key, out);
		ListElem e(key, out);
		l.push_front(e);
		m.emplace(key, l.begin());
	}

	// convenience function
	template <class Func>
	Val get(const Key &key, Func getVal) {
		Val out;
		get(key, out);
		return out;
	}
};
